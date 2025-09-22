"""Pattern definitions for consciousness analysis"""

# Complexity domain keywords
DOMAIN_PATTERNS = {
    'healthcare': {
        'keywords': ['insurance', 'claim', 'denied', 'appeal', 'prior auth',
                    'coverage', 'deductible', 'copay', 'formulary', 'medical bill',
                    'ADHD', 'medication', 'pharmacy', 'doctor', 'specialist'],
        'time_violence_indicators': ['on hold', 'hours', 'waiting', 'finally',
                                    'months', 'fighting', 'exhausted'],
        'consciousness_indicators': ['realized', 'pattern', 'always', 'every time',
                                    'figured out', 'trick is', 'learned']
    },
    'financial': {
        'keywords': ['debt', 'bankruptcy', 'collections', 'credit', 'loan',
                    'student loans', 'interest', 'payment plan', 'foreclosure',
                    'eviction', 'overdraft', 'bank'],
        'time_violence_indicators': ['paperwork', 'documents', 'forms', 'calls',
                                    'letters', 'notices', 'deadlines'],
        'consciousness_indicators': ['systemic', 'rigged', 'designed to', 'trap',
                                    'escape', 'beat the system', 'loophole']
    },
    'employment': {
        'keywords': ['job search', 'interview', 'resume', 'ATS', 'application',
                    'rejection', 'ghosted', 'salary', 'negotiation', 'linkedin',
                    'unemployed', 'laid off', 'fired'],
        'time_violence_indicators': ['applications', 'rounds', 'months looking',
                                    'never heard back', 'automated rejection'],
        'consciousness_indicators': ['game', 'algorithm', 'keyword', 'hack',
                                    'actually works', 'real secret']
    },
    'government': {
        'keywords': ['DMV', 'IRS', 'tax', 'benefits', 'disability', 'SSA',
                    'SNAP', 'welfare', 'unemployment', 'permits', 'license'],
        'time_violence_indicators': ['queue', 'line', 'website down', 'office',
                                    'appointment', 'weeks wait', 'processing'],
        'consciousness_indicators': ['bureaucracy', 'kafkaesque', 'catch-22',
                                    'workaround', 'backdoor', 'right person']
    },
    'education': {
        'keywords': ['FAFSA', 'financial aid', 'tuition', 'transcript', 'credits',
                    'registration', 'advisor', 'requirements', 'prerequisite',
                    'degree', 'graduation'],
        'time_violence_indicators': ['runaround', 'different answers', 'conflicting',
                                    'no one knows', 'sent between departments'],
        'consciousness_indicators': ['navigate', 'system works', 'hidden requirement',
                                    'unwritten rule', 'actually need']
    }
}

# Neurodivergent indicators (often correlated with consciousness workers)
NEURODIVERGENT_PATTERNS = [
    'ADHD', 'autism', 'autistic', 'AuDHD', 'neurodivergent', 'ND',
    'executive dysfunction', 'rejection sensitive', 'RSD', 'hyperfocus',
    'stimming', 'masking', 'burnout', 'meltdown', 'shutdown',
    'sensory', 'overwhelm', 'spoons', 'dysregulation'
]

# Bio analysis keywords for consciousness indicators
BIO_KEYWORDS = {
    'neurodivergent_terms': ['adhd', 'autistic', 'neurodivergent', 'nd', 'audhd', 'actually autistic'],
    'fighter_terms': ['advocate', 'activist', 'survivor', 'fighter', 'reformer'],
    'domain_expertise': {
        'healthcare': ['patient advocate', 'medical', 'health', 'insurance fighter'],
        'financial': ['debt free', 'bankruptcy survivor', 'financial literacy'],
        'employment': ['job seeker', 'career coach', 'unemployed', 'laid off'],
        'education': ['student', 'grad school', 'student loans', 'dropout']
    }
}

# Consciousness type classifications
CONSCIOUSNESS_TYPES = {
    'battle': ['fighting', 'battle', 'struggle', 'dealing'],
    'pattern': ['realized', 'pattern', 'always', 'every'],
    'solution': ['solution', 'fixed', 'solved', 'hack'],
    'liberation': ['free', 'escaped', 'done', 'over'],
    'awareness': []  # Default if no other type matches
}

# Time extraction patterns
TIME_PATTERNS = {
    'hours': r'(\d+)\s*hours?',
    'hours_abbr': r'(\d+)\s*hrs?',
    'minutes': r'(\d+)\s*minutes?',
    'minutes_abbr': r'(\d+)\s*mins?',
    'days': r'(\d+)\s*days?',
    'weeks': r'(\d+)\s*weeks?',
    'months': r'(\d+)\s*months?'
}

# Time conversion factors (to hours)
TIME_CONVERSIONS = {
    'minutes': 1/60,
    'minutes_abbr': 1/60,
    'hours': 1,
    'hours_abbr': 1,
    'days': 8,  # Assume 8 hours per day of dealing with complexity
    'weeks': 20,  # Assume 20 hours per week
    'months': 80  # Assume 80 hours per month
}

# Implicit time indicators
IMPLICIT_TIME_INDICATORS = {
    'all_day': ['all day', 'entire day', 'whole day'],
    'half_day': ['all morning', 'all afternoon'],
    'significant': ['finally', 'eventually', 'at last']
}

IMPLICIT_TIME_VALUES = {
    'all_day': 8,
    'half_day': 4,
    'significant': 2
}

# Profanity indicators (for intensity calculation)
PROFANITY_INDICATORS = ['fuck', 'shit', 'damn', 'hell', 'bullshit', 'wtf']