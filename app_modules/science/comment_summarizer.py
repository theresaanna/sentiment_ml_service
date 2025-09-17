"""
Comment Summarization using AI models.
"""
import os
import logging
from typing import List, Dict, Optional
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import openai
from app.cache import cache

logger = logging.getLogger(__name__)


class CommentSummarizer:
    """AI-powered comment summarizer for YouTube video comments."""
    
    def __init__(self, use_openai: bool = False):
        """
        Initialize the comment summarizer.
        
        Args:
            use_openai: Whether to use OpenAI API for summarization
        """
        self.use_openai = use_openai and os.getenv('OPENAI_API_KEY')
        
        if self.use_openai:
            openai.api_key = os.getenv('OPENAI_API_KEY')
            logger.info("Using OpenAI for comment summarization")
        else:
            # Use open-source summarization model
            logger.info("Loading BART model for comment summarization...")
            self.model_name = "facebook/bart-large-cnn"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.summarizer = pipeline(
                "summarization",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if os.environ.get('CUDA_VISIBLE_DEVICES') else -1
            )
    
    def _prepare_comments_text(self, comments: List[Dict], sentiment_results: Optional[Dict] = None) -> str:
        """
        Prepare comments for summarization.
        
        Args:
            comments: List of comment dictionaries
            sentiment_results: Optional sentiment analysis results
            
        Returns:
            Formatted text for summarization
        """
        # Group comments by sentiment if available
        if sentiment_results and 'individual_results' in sentiment_results:
            positive_comments = []
            negative_comments = []
            neutral_comments = []
            
            for i, comment in enumerate(comments[:len(sentiment_results['individual_results'])]):
                sentiment = sentiment_results['individual_results'][i]['predicted_sentiment']
                text = comment.get('text', '')
                
                if sentiment == 'positive':
                    positive_comments.append(text)
                elif sentiment == 'negative':
                    negative_comments.append(text)
                else:
                    neutral_comments.append(text)
            
            # Create balanced text representation
            text_parts = []
            
            if positive_comments:
                text_parts.append(f"Supportive viewpoints: {' '.join(positive_comments[:10])}")
            if negative_comments:
                text_parts.append(f"Critical perspectives: {' '.join(negative_comments[:10])}")
            if neutral_comments:
                text_parts.append(f"Neutral observations: {' '.join(neutral_comments[:10])}")
            
            combined_text = " ".join(text_parts)
        else:
            # Simple concatenation if no sentiment data
            combined_text = " ".join([c.get('text', '') for c in comments[:30]])
        
        # Truncate to reasonable length
        max_length = 4000
        if len(combined_text) > max_length:
            combined_text = combined_text[:max_length]
        
        return combined_text
    
    def summarize_with_openai(self, comments: List[Dict], sentiment_results: Optional[Dict] = None) -> Dict:
        """
        Generate summary using OpenAI API.
        
        Args:
            comments: List of comment dictionaries
            sentiment_results: Optional sentiment analysis results
            
        Returns:
            Summary dictionary
        """
        try:
            # Prepare prompt
            comments_text = self._prepare_comments_text(comments, sentiment_results)
            
            # Add sentiment context if available
            sentiment_context = ""
            if sentiment_results:
                sentiment_context = f"""
                Overall sentiment: {sentiment_results.get('overall_sentiment', 'Unknown')}
                Positive: {sentiment_results.get('sentiment_percentages', {}).get('positive', 0):.1f}%
                Negative: {sentiment_results.get('sentiment_percentages', {}).get('negative', 0):.1f}%
                Neutral: {sentiment_results.get('sentiment_percentages', {}).get('neutral', 0):.1f}%
                """
            
            prompt = f"""
            Analyze and summarize these YouTube video comments in a conversational, engaging style. Make it verbose and insightful, like you're explaining the comment section to a friend.
            
            {sentiment_context}
            
            Comments:
            {comments_text}
            
            Write a comprehensive analysis that:
            
            1. **Opens conversationally** - Start with how the audience is reacting ("The audience is absolutely loving this!" or "This content has really split viewers down the middle" etc.)
            
            2. **Discusses major themes** - What are people actually talking about? Use phrases like:
               - "Looking at what people are actually talking about..."
               - "There's quite a bit of discussion around..."
               - "Many viewers are getting into..."
               - "A big talking point seems to be..."
            
            3. **Identifies emotional undercurrents** - Go beyond basic sentiment:
               - "Emotionally, the comment section shows..."
               - "There's genuine enthusiasm here..."
               - "You can sense some frustration when viewers mention..."
               - "People seem genuinely grateful for..."
            
            4. **Spots controversies and debates** - If there are heated discussions:
               - "🔥 There are some heated discussions brewing..."
               - "We're seeing viewers debate and disagree about..."
               - "Tensions are running high around..."
            
            5. **Analyzes engagement patterns**:
               - "The comment section shows thoughtful, detailed responses..."
               - "Most comments are quick reactions rather than deep analysis..."
               - "There's active discussion with people asking questions..."
            
            6. **Ends with key insights** using emojis:
               - "💡 This appears to be content that genuinely helps people..."
               - "📊 The high confidence in sentiment suggests people are being direct..."
               - "🎯 This has hit a nerve and created real division..."
            
            Make it feel like a smart friend is walking you through what's happening in the comments. Be specific about what people are saying, but maintain balance. Use conversational language with strategic emojis for emphasis.
            
            Aim for 3-4 substantial paragraphs that give real insight into the comment section dynamics.
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an insightful social media analyst with a conversational, engaging style. You excel at reading between the lines of comment sections to understand what's really happening. You write like you're explaining the dynamics to a smart friend - informative but accessible, thorough but engaging. You use strategic emojis for emphasis and maintain balance while being genuinely insightful about human behavior and digital discourse."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,  # Allow for longer, more detailed summaries
                temperature=0.7
            )
            
            summary_text = response.choices[0].message.content
            
            return {
                'summary': summary_text,
                'method': 'openai',
                'comments_analyzed': len(comments)
            }
            
        except Exception as e:
            logger.error(f"Error using OpenAI API: {e}")
            # Fallback to local model
            return self.create_objective_summary(comments, sentiment_results)
    
    def summarize_with_transformer(self, comments: List[Dict], sentiment_results: Optional[Dict] = None) -> Dict:
        """
        Generate summary using local transformer model.
        
        Args:
            comments: List of comment dictionaries
            sentiment_results: Optional sentiment analysis results
            
        Returns:
            Summary dictionary
        """
        try:
            # Prepare text
            comments_text = self._prepare_comments_text(comments, sentiment_results)
            
            # Generate summary
            summary = self.summarizer(
                comments_text,
                max_length=150,
                min_length=50,
                do_sample=False
            )
            
            summary_text = summary[0]['summary_text']
            
            # Add objective sentiment insights if available
            if sentiment_results:
                pos_pct = sentiment_results.get('sentiment_percentages', {}).get('positive', 0)
                neg_pct = sentiment_results.get('sentiment_percentages', {}).get('negative', 0)
                neu_pct = sentiment_results.get('sentiment_percentages', {}).get('neutral', 0)
                
                sentiment_insight = f"\n\nViewer Response Distribution: "
                
                # Create objective framing based on the distribution
                if pos_pct > neg_pct + 10:
                    sentiment_insight += f"The majority of viewers ({pos_pct:.1f}%) expressed positive reactions, while {neg_pct:.1f}% shared concerns or criticisms."
                elif neg_pct > pos_pct + 10:
                    sentiment_insight += f"A significant portion of viewers ({neg_pct:.1f}%) expressed concerns or criticisms, while {pos_pct:.1f}% shared positive reactions."
                else:
                    sentiment_insight += f"Viewers showed mixed reactions, with {pos_pct:.1f}% positive, {neg_pct:.1f}% negative, and {neu_pct:.1f}% neutral responses, indicating diverse opinions on the content."
                
                summary_text += sentiment_insight
            
            return {
                'summary': summary_text,
                'method': 'transformer',
                'comments_analyzed': len(comments)
            }
            
        except Exception as e:
            logger.error(f"Error generating summary with transformer: {e}")
            return {
                'summary': "Unable to generate summary at this time.",
                'method': 'error',
                'error': str(e)
            }
    
    def generate_summary(self, comments: List[Dict], sentiment_results: Optional[Dict] = None) -> Dict:
        """
        Generate a comprehensive summary of comments.
        
        Args:
            comments: List of comment dictionaries
            sentiment_results: Optional sentiment analysis results
            
        Returns:
            Dictionary containing summary and metadata
        """
        if not comments:
            return {
                'summary': "No comments available to summarize.",
                'method': 'none',
                'comments_analyzed': 0
            }
        
        # Check cache first
        cache_key = f"comment_summary:{hash(str(comments[:10]))}"
        cached_summary = cache.get('comment_summary', cache_key)
        if cached_summary:
            logger.info("Using cached comment summary")
            return cached_summary
        
        # Generate summary
        if self.use_openai:
            result = self.summarize_with_openai(comments, sentiment_results)
        else:
            result = self.create_objective_summary(comments, sentiment_results)
        
        # Extract key themes
        result['key_themes'] = self._extract_themes(comments)
        
        # Add engagement metrics
        result['engagement_metrics'] = self._calculate_engagement_metrics(comments)
        
        # Cache the result
        cache.set('comment_summary', cache_key, result, ttl_hours=6)  # Cache for 6 hours
        
        return result
    
    def create_objective_summary(self, comments: List[Dict], sentiment_results: Optional[Dict] = None) -> Dict:
        """
        Create an objective, balanced summary without using complex AI models.
        
        Args:
            comments: List of comment dictionaries
            sentiment_results: Optional sentiment analysis results
            
        Returns:
            Summary dictionary with objective analysis
        """
        try:
            if not comments:
                return {
                    'summary': "No comments available to analyze.",
                    'method': 'objective',
                    'comments_analyzed': 0
                }
            
            # Categorize comments by sentiment
            positive_comments = []
            negative_comments = []
            neutral_comments = []
            
            if sentiment_results and 'individual_results' in sentiment_results:
                for i, comment in enumerate(comments[:len(sentiment_results['individual_results'])]):
                    sentiment = sentiment_results['individual_results'][i]['predicted_sentiment']
                    text = comment.get('text', '').strip()
                    
                    if text and len(text) > 10:  # Only include substantial comments
                        if sentiment == 'positive':
                            positive_comments.append(text)
                        elif sentiment == 'negative':
                            negative_comments.append(text)
                        else:
                            neutral_comments.append(text)
            else:
                # If no sentiment data, just use all comments as neutral
                neutral_comments = [c.get('text', '') for c in comments if c.get('text', '').strip()]
            
            # Extract key themes and topics
            themes = self._extract_themes(comments, top_n=5)
            
            # Build objective summary
            summary_parts = []
            
            # Overview based on sentiment distribution
            total_comments = len(positive_comments) + len(negative_comments) + len(neutral_comments)
            if total_comments == 0:
                return {
                    'summary': "The comments section contains minimal substantive discussion.",
                    'method': 'objective',
                    'comments_analyzed': len(comments)
                }
            
            pos_pct = (len(positive_comments) / total_comments) * 100
            neg_pct = (len(negative_comments) / total_comments) * 100
            neu_pct = (len(neutral_comments) / total_comments) * 100
            
            # Create balanced opening
            if pos_pct > neg_pct + 15:
                summary_parts.append(f"The video generated predominantly positive responses from viewers ({pos_pct:.0f}% positive), with some critical perspectives ({neg_pct:.0f}% negative) also represented.")
            elif neg_pct > pos_pct + 15:
                summary_parts.append(f"The video sparked significant debate, with many viewers expressing concerns or criticisms ({neg_pct:.0f}% negative), while others offered supportive viewpoints ({pos_pct:.0f}% positive).")
            else:
                summary_parts.append(f"The video generated diverse reactions from viewers, with opinions fairly divided between supportive ({pos_pct:.0f}% positive) and critical ({neg_pct:.0f}% negative) perspectives.")
            
            # Add perspective details
            perspective_details = []
            
            # Analyze positive themes
            if positive_comments:
                positive_themes = self._identify_common_themes(positive_comments[:5])
                if positive_themes:
                    perspective_details.append(f"Some viewers appreciated aspects such as {', '.join(positive_themes[:3]).lower()}.")
            
            # Analyze negative themes
            if negative_comments:
                negative_themes = self._identify_common_themes(negative_comments[:5])
                if negative_themes:
                    perspective_details.append(f"Others raised concerns about {', '.join(negative_themes[:3]).lower()}.")
            
            # Add main discussion topics
            if themes:
                perspective_details.append(f"Key topics in the discussion included {', '.join(themes[:3]).lower()}.")
            
            # Combine all parts
            if perspective_details:
                summary_parts.extend(perspective_details)
            
            # Add engagement context
            engagement_context = self._get_engagement_context(comments)
            if engagement_context:
                summary_parts.append(engagement_context)
            
            final_summary = " ".join(summary_parts)
            
            return {
                'summary': final_summary,
                'method': 'objective',
                'comments_analyzed': len(comments)
            }
            
        except Exception as e:
            logger.error(f"Error creating objective summary: {e}")
            return {
                'summary': "Unable to generate summary due to processing error.",
                'method': 'objective_error',
                'error': str(e),
                'comments_analyzed': len(comments)
            }
    
    def _identify_common_themes(self, comment_texts: List[str]) -> List[str]:
        """Identify common themes in a subset of comments."""
        from collections import Counter
        import re
        
        # Key terms that might indicate themes
        theme_keywords = {
            'feminism': ['feminist', 'feminism', 'equality', 'women', 'gender'],
            'performance': ['performance', 'acting', 'talent', 'skill'],
            'message': ['message', 'point', 'statement', 'meaning'],
            'interview': ['interview', 'interviewer', 'questions', 'conversation'],
            'authenticity': ['authentic', 'real', 'genuine', 'honest'],
            'controversy': ['controversial', 'debate', 'argument', 'disagreement'],
            'style': ['style', 'fashion', 'look', 'appearance'],
            'music': ['music', 'song', 'album', 'artist']
        }
        
        theme_counts = Counter()
        
        for text in comment_texts:
            text_lower = text.lower()
            for theme, keywords in theme_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    theme_counts[theme] += 1
        
        # Return themes found in multiple comments
        return [theme for theme, count in theme_counts.most_common(3) if count > 1]
    
    def _get_engagement_context(self, comments: List[Dict]) -> str:
        """Generate context about comment engagement."""
        if not comments:
            return ""
        
        # Calculate basic engagement metrics
        total_likes = sum(c.get('likes', 0) for c in comments)
        avg_likes = total_likes / len(comments) if comments else 0
        replies = sum(1 for c in comments if c.get('is_reply', False))
        
        if avg_likes > 10:
            return "The discussion generated significant engagement, with many comments receiving multiple likes."
        elif replies > len(comments) * 0.3:
            return "The topic prompted active discussion with numerous replies and responses."
        else:
            return "The video generated a moderate level of viewer engagement and discussion."
    
    def _extract_themes(self, comments: List[Dict], top_n: int = 5) -> List[str]:
        """
        Extract key themes from comments.
        
        Args:
            comments: List of comment dictionaries
            top_n: Number of top themes to return
            
        Returns:
            List of key themes
        """
        from collections import Counter
        import re
        
        # Common words to exclude
        stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                         'to', 'for', 'of', 'with', 'by', 'from', 'is', 'was', 
                         'are', 'were', 'been', 'be', 'have', 'has', 'had', 
                         'do', 'does', 'did', 'will', 'would', 'could', 'should',
                         'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
                         'it', 'we', 'they', 'what', 'which', 'who', 'when', 'where',
                         'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
                         'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
                         'own', 'same', 'so', 'than', 'too', 'very', 'can', 'just'])
        
        # Extract words from all comments
        all_words = []
        for comment in comments:
            text = comment.get('text', '').lower()
            # Extract words (alphanumeric only)
            words = re.findall(r'\b[a-z]+\b', text)
            # Filter out stop words and short words
            words = [w for w in words if len(w) > 3 and w not in stop_words]
            all_words.extend(words)
        
        # Count word frequencies
        word_counts = Counter(all_words)
        
        # Get top themes
        top_themes = [word for word, count in word_counts.most_common(top_n)]
        
        return top_themes
    
    def _calculate_engagement_metrics(self, comments: List[Dict]) -> Dict:
        """
        Calculate engagement metrics from comments.
        
        Args:
            comments: List of comment dictionaries
            
        Returns:
            Dictionary of engagement metrics
        """
        # Use 'likes' field instead of 'like_count' as per the actual YouTube API response
        total_likes = sum(c.get('likes', 0) for c in comments)
        avg_likes = total_likes / len(comments) if comments else 0
        
        # Find most liked comment
        most_liked = max(comments, key=lambda c: c.get('likes', 0)) if comments else None
        
        # Calculate reply rate
        replies = sum(1 for c in comments if c.get('is_reply', False))
        reply_rate = (replies / len(comments) * 100) if comments else 0
        
        return {
            'total_likes': total_likes,
            'average_likes': round(avg_likes, 2),
            'most_liked_comment': most_liked.get('text', '')[:200] if most_liked else '',
            'most_liked_count': most_liked.get('likes', 0) if most_liked else 0,
            'reply_rate': round(reply_rate, 2)
        }
