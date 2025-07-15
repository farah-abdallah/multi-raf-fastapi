"""
Clear the analytics database to remove old records with incorrect faithfulness scores
"""
import sqlite3
import os

def clear_database():
    """Clear the evaluation database"""
    db_path = "rag_evaluation.db"
    
    if not os.path.exists(db_path):
        print("ℹ️ Database doesn't exist yet - nothing to clear")
        return
    
    try:
        # Connect and clear
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Count records before deletion
        cursor.execute("SELECT COUNT(*) FROM evaluations")
        records_before = cursor.fetchone()[0]
        
        if records_before == 0:
            print("ℹ️ Database is already empty")
            conn.close()
            return
        
        # Delete all records
        cursor.execute("DELETE FROM evaluations")
        conn.commit()
        
        # Verify deletion
        cursor.execute("SELECT COUNT(*) FROM evaluations")
        records_after = cursor.fetchone()[0]
        
        conn.close()
        
        print(f"✅ Successfully cleared database!")
        print(f"📊 Removed {records_before} old records")
        print(f"🔍 Current record count: {records_after}")
        
    except Exception as e:
        print(f"❌ Error clearing database: {e}")

if __name__ == "__main__":
    print("🧹 Clearing analytics database...")
    clear_database()
    print("🚀 Ready for fresh analytics data!")
