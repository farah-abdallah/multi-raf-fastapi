"""Check current database content to understand why faithfulness is still 0"""

import sqlite3
import json
from datetime import datetime

def check_database_content():
    """Check what's currently in the evaluation database"""
    
    db_path = "rag_evaluation.db"
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all evaluations
        cursor.execute("""
            SELECT id, technique, query, context, faithfulness_score, created_at 
            FROM evaluations 
            ORDER BY created_at DESC 
            LIMIT 5
        """)
        
        results = cursor.fetchall()
        
        print("🔍 Current Database Content (Last 5 records):")
        print("=" * 80)
        
        for row in results:
            id, technique, query, context, faithfulness, created_at = row
            print(f"📊 Record ID: {id}")
            print(f"🔧 Technique: {technique}")
            print(f"❓ Query: {query[:50]}...")
            print(f"📝 Context: {context[:100] if context else 'None'}...")
            print(f"📊 Faithfulness Score: {faithfulness}")
            print(f"📅 Created: {created_at}")
            print("-" * 40)
        
        # Check context patterns
        cursor.execute("SELECT DISTINCT context FROM evaluations WHERE context IS NOT NULL")
        contexts = cursor.fetchall()
        
        print("\n🔍 Unique Context Patterns:")
        print("=" * 50)
        for ctx in contexts[:5]:  # Show first 5 unique contexts
            context_str = str(ctx[0])[:100]
            print(f"• {context_str}...")
        
        # Check faithfulness distribution
        cursor.execute("SELECT technique, AVG(faithfulness_score), COUNT(*) FROM evaluations GROUP BY technique")
        stats = cursor.fetchall()
        
        print("\n📊 Faithfulness Statistics by Technique:")
        print("=" * 50)
        for technique, avg_faith, count in stats:
            print(f"• {technique}: Avg={avg_faith:.3f}, Count={count}")
        
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking database: {e}")
        return False

if __name__ == "__main__":
    check_database_content()
