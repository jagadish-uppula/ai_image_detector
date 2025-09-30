import numpy as np
from django.db.models import Count, Avg, Q
from datetime import datetime, timedelta
from detector.models import AnalysisHistory

def get_visualization_data(user):
    """Get all visualization data for the user"""
    try:
        # Get all analyses for the user
        analyses = AnalysisHistory.objects.filter(user=user)
        
        # Basic counts
        total_analyses = analyses.count()
        real_images = analyses.filter(is_ai_generated=False).count()
        ai_generated = analyses.filter(is_ai_generated=True).count()
        avg_confidence = analyses.aggregate(avg_confidence=Avg('confidence'))['avg_confidence'] or 0
        
        # Confidence distribution (10 bins from 0-100)
        confidence_bins = np.linspace(0, 100, 11)
        real_counts = []
        ai_counts = []
        
        for i in range(len(confidence_bins)-1):
            lower = confidence_bins[i]
            upper = confidence_bins[i+1]
            
            real_counts.append(analyses.filter(
                is_ai_generated=False,
                confidence__gte=lower,
                confidence__lt=upper
            ).count())
            
            ai_counts.append(analyses.filter(
                is_ai_generated=True,
                confidence__gte=lower,
                confidence__lt=upper
            ).count())
        
        # Similarity data (convert to percentage)
        real_similarities = list(analyses.filter(
            is_ai_generated=False,
            similarity_score__isnull=False
        ).values_list('similarity_score', flat=True)) or [0]
        
        ai_similarities = list(analyses.filter(
            is_ai_generated=True,
            similarity_score__isnull=False
        ).values_list('similarity_score', flat=True)) or [0]
        
        # Timeline data (last 7 days)
        dates = [(datetime.now() - timedelta(days=i)).date() for i in range(6, -1, -1)]
        timeline_labels = [date.strftime('%Y-%m-%d') for date in dates]
        
        real_timeline = []
        ai_timeline = []
        
        for date in dates:
            real_timeline.append(analyses.filter(
                created_at__date=date,
                is_ai_generated=False
            ).count())
            ai_timeline.append(analyses.filter(
                created_at__date=date,
                is_ai_generated=True
            ).count())
        
        return {
            'total_analyses': total_analyses,
            'real_images': real_images,
            'ai_generated': ai_generated,
            'avg_confidence': avg_confidence,
            'confidence_data': {
                'real': real_counts,
                'ai': ai_counts,
                'labels': [f"{int(b)}-{int(b)+10}%" for b in confidence_bins[:-1]]
            },
            'similarity_data': {
                'real': [score * 100 for score in real_similarities],  # Convert to percentage
                'ai': [score * 100 for score in ai_similarities]      # Convert to percentage
            },
            'timeline_data': {
                'labels': timeline_labels,
                'real': real_timeline,
                'ai': ai_timeline
            }
        }
    except Exception as e:
        print(f"Error in get_visualization_data: {str(e)}")
        # Return default values if there's an error
        return {
            'total_analyses': 0,
            'real_images': 0,
            'ai_generated': 0,
            'avg_confidence': 0,
            'confidence_data': {
                'real': [0] * 10,
                'ai': [0] * 10,
                'labels': [f"{i*10}-{(i+1)*10}%" for i in range(10)]
            },
            'similarity_data': {
                'real': [0],
                'ai': [0]
            },
            'timeline_data': {
                'labels': [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(6, -1, -1)],
                'real': [0] * 7,
                'ai': [0] * 7
            }
        }