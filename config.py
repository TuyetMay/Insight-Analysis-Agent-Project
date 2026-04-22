import os

class Config:
    """Application configuration"""
    
    # Database settings
    DB_HOST = os.getenv('DB_HOST', 'db.yihxtwvezbdgcczwepuj.supabase.co')
    DB_PORT = os.getenv('DB_PORT', '5432')
    DB_NAME = os.getenv('DB_NAME', 'postgres')
    DB_USER = os.getenv('DB_USER', 'postgres')
    DB_PASSWORD = os.getenv('DB_PASSWORD', '')
    DB_TABLE = os.getenv('DB_TABLE', 'superstore')
    
    # App settings
    APP_TITLE = os.getenv('APP_TITLE', 'Superstore Business Intelligence Dashboard')
    APP_ICON = os.getenv('APP_ICON', '📊')

    # Vertex AI settings – uses your $300 GCP free credit
    GCP_PROJECT  = os.getenv("GCP_PROJECT", "")
    GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-002")

    # Legacy AI Studio key – kept for backward-compat checks; no longer used
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

    
    @classmethod
    def get_db_connection_string(cls):
        """Generate database connection string"""
        return f"host={cls.DB_HOST} port={cls.DB_PORT} dbname={cls.DB_NAME} user={cls.DB_USER} password={cls.DB_PASSWORD}"

