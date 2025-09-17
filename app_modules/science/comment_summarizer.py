"""
Comment Summarization using OpenAI.
"""
import os
import logging
from typing import List, Dict, Optional
from openai import OpenAI
from cache import cache

logger = logging.getLogger(__name__)


class CommentSummarizer:
    """OpenAI-powered comment summarizer for YouTube video comments."""
    
    def __init__(self):
        """
        Initialize the comment summarizer with OpenAI.
        """
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        # Optional overrides
        self.model_name = os.getenv('OPENAI_SUMMARY_MODEL', 'gpt-4o-mini')
        timeout_s = float(os.getenv('OPENAI_TIMEOUT_SECONDS', '30'))

        # Initialize client with timeout
        self.client = OpenAI(api_key=openai_key, timeout=timeout_s)
        logger.info(f"Using OpenAI for comment summarization (model={self.model_name})")
    
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
    
    def generate_summary(self, comments: List[Dict], sentiment_results: Optional[Dict] = None) -> Dict:
        """
        Generate a comprehensive summary of comments using OpenAI.
        
        Args:
            comments: List of comment dictionaries
            sentiment_results: Optional sentiment analysis results
            
        Returns:
            Dictionary containing summary and metadata
        """
        if not comments:
            return {
                'summary': "No comments available to summarize.",
                'method': 'openai',
                'comments_analyzed': 0
            }
        
        # Check cache first
        cache_key = f"comment_summary:{hash(str(comments[:10]))}"
        cached_summary = cache.get('comment_summary', cache_key)
        if cached_summary:
            logger.info("Using cached comment summary")
            return cached_summary
        
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
            
            Additionally, highlight commonalities and differences between the themes discussed. Look for patterns where viewers agree or disagree, and describe these dynamics in detail. Make the summary a substantial paragraph that really digs into the social dynamics at play.
            
            Make it feel like a smart friend is walking you through what's happening in the comments. Be specific about what people are saying, but maintain balance. Use conversational language with strategic emojis for emphasis.
            
            Aim for 3-4 substantial paragraphs that give real insight into the comment section dynamics.
            """
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an insightful social media analyst with a conversational, engaging style. You excel at reading between the lines of comment sections to understand what's really happening. You write like you're explaining the dynamics to a smart friend - informative but accessible, thorough but engaging. You use strategic emojis for emphasis and maintain balance while being genuinely insightful about human behavior and digital discourse."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,  # Allow for longer, more detailed summaries
                temperature=0.7
            )
            
            summary_text = response.choices[0].message.content
            
            result = {
                'summary': summary_text,
                'method': 'openai',
                'comments_analyzed': len(comments)
            }
            
            # Extract key themes
            result['key_themes'] = self._extract_themes(comments)
            
            # Add engagement metrics
            result['engagement_metrics'] = self._calculate_engagement_metrics(comments)
            
            # Cache the result
            cache.set('comment_summary', cache_key, result, ttl_hours=6)
            
            return result
            
        except Exception as e:
            err_type = type(e).__name__
            logger.error(f"Error using OpenAI API: {err_type}: {e}")
            return {
                'summary': f"Unable to generate summary due to OpenAI API error ({err_type}).",
                'method': 'openai_error',
                'error': f"{err_type}: {str(e)}",
                'comments_analyzed': len(comments)
            }
    
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
