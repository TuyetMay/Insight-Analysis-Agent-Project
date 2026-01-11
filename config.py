import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration"""
    
    # Database settings
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = os.getenv('DB_PORT', '5432')
    DB_NAME = os.getenv('DB_NAME', 'superstore')
    DB_USER = os.getenv('DB_USER', 'nguyenthituyetmay')
    DB_PASSWORD = os.getenv('DB_PASSWORD', '')
    DB_TABLE = os.getenv('DB_TABLE', 'superstore')
    
    # App settings
    APP_TITLE = os.getenv('APP_TITLE', 'Superstore Business Intelligence Dashboard')
    APP_ICON = os.getenv('APP_ICON', '📊')

    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-lite")

    
    @classmethod
    def get_db_connection_string(cls):
        """Generate database connection string"""
        return f"host={cls.DB_HOST} port={cls.DB_PORT} dbname={cls.DB_NAME} user={cls.DB_USER} password={cls.DB_PASSWORD}"

