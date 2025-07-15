"""
Analytics Dashboard for RAG Evaluation

This module provides visualization and analytics components for the Streamlit app
to display RAG technique performance comparisons and insights.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import time

def create_performance_overview(comparison_data: Dict[str, Any]) -> None:
    """Create overview cards showing key performance metrics"""
    if not comparison_data:
        st.warning("No evaluation data available yet. Start chatting to see analytics!")
        return
      # Convert to DataFrame for easier manipulation
    df = pd.DataFrame.from_dict(comparison_data, orient='index')
    
    # Calculate overall statistics, handling None values
    total_queries = df['total_queries'].sum() if 'total_queries' in df.columns else 0
    avg_processing_time = df['avg_processing_time'].mean() if 'avg_processing_time' in df.columns else None
    avg_user_rating = df['avg_user_rating'].mean() if 'avg_user_rating' in df.columns else None
    
    # Display overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Total Queries",
            value=f"{int(total_queries)}",
            help="Total number of queries evaluated across all techniques"
        )
    
    with col2:
        st.metric(
            label="⚡ Avg Processing Time",
            value=f"{avg_processing_time:.2f}s" if pd.notna(avg_processing_time) else "N/A",
            help="Average time taken to process queries"
        )
    
    with col3:
        st.metric(
            label="⭐ Avg User Rating",
            value=f"{avg_user_rating:.1f}/5" if pd.notna(avg_user_rating) else "N/A",
            help="Average user satisfaction rating"
        )
    
    with col4:
        # Handle case where avg_user_rating column doesn't exist or is all NaN
        if 'avg_user_rating' in df.columns and not df['avg_user_rating'].isna().all():
            best_technique = df.loc[df['avg_user_rating'].idxmax()]
            st.metric(
                label="🏆 Top Rated Technique",
                value=best_technique.name,
                help="Technique with highest user rating"
            )
        else:
            st.metric(
                label="🏆 Top Rated Technique",
                value="N/A",
                help="No ratings available yet"
            )

def create_technique_comparison_chart(comparison_data: Dict[str, Any]) -> None:
    """Create radar chart comparing RAG techniques"""
    if not comparison_data:
        return
    
    st.subheader("🕸️ Technique Performance Radar")
    
    # Prepare data for radar chart
    techniques = list(comparison_data.keys())
    metrics = ['avg_relevance', 'avg_faithfulness', 'avg_completeness', 'avg_semantic_similarity']
    metric_labels = ['Relevance', 'Faithfulness', 'Completeness', 'Semantic Similarity']
    
    fig = go.Figure()
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    for i, technique in enumerate(techniques):
        values = []
        for metric in metrics:
            val = comparison_data[technique].get(metric, 0)
            values.append(val if val is not None else 0)
        
        # Close the radar chart
        values_closed = values + [values[0]]
        labels_closed = metric_labels + [metric_labels[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=labels_closed,
            fill='toself',
            name=technique,
            line_color=colors[i % len(colors)],
            fillcolor=colors[i % len(colors)],
            opacity=0.3
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="RAG Technique Performance Comparison",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_performance_metrics_chart(comparison_data: Dict[str, Any]) -> None:
    """Create bar charts for different performance metrics"""
    if not comparison_data:
        return
    
    st.subheader("📊 Performance Metrics Breakdown")
    
    df = pd.DataFrame.from_dict(comparison_data, orient='index')
    
    # Fill None values with 0 for visualization
    df = df.fillna(0)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Relevance Scores', 'Processing Time', 'User Ratings', 'Response Length'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    techniques = df.index.tolist()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    # Relevance scores
    if 'avg_relevance' in df.columns:
        fig.add_trace(
            go.Bar(x=techniques, y=df['avg_relevance'], name='Relevance', 
                   marker_color=[colors[i % len(colors)] for i in range(len(techniques))]),
            row=1, col=1
        )
    
    # Processing time
    if 'avg_processing_time' in df.columns:
        fig.add_trace(
            go.Bar(x=techniques, y=df['avg_processing_time'], name='Processing Time',
                   marker_color=[colors[i % len(colors)] for i in range(len(techniques))]),
            row=1, col=2
        )
    
    # User ratings
    if 'avg_user_rating' in df.columns:
        fig.add_trace(
            go.Bar(x=techniques, y=df['avg_user_rating'], name='User Rating',
                   marker_color=[colors[i % len(colors)] for i in range(len(techniques))]),
            row=2, col=1
        )
    
    # Response length
    if 'avg_response_length' in df.columns:
        fig.add_trace(
            go.Bar(x=techniques, y=df['avg_response_length'], name='Response Length',
                   marker_color=[colors[i % len(colors)] for i in range(len(techniques))]),
            row=2, col=2
        )
    
    fig.update_layout(height=600, showlegend=False, title_text="Performance Metrics by Technique")
    fig.update_xaxes(tickangle=45)
    
    st.plotly_chart(fig, use_container_width=True)

def create_time_series_chart(performance_history: List[Dict]) -> None:
    """Create time series chart showing performance over time"""
    if not performance_history:
        return
    
    st.subheader("📈 Performance Trends Over Time")
    
    # Convert to DataFrame
    df = pd.DataFrame(performance_history)
    df['created_at'] = pd.to_datetime(df['created_at'])
    
    # Group by technique and date
    df_daily = df.groupby([df['created_at'].dt.date, 'technique']).agg({
        'relevance_score': 'mean',
        'overall_rating': 'mean',
        'processing_time': 'mean'
    }).reset_index()
    
    # Create tabs for different metrics
    tab1, tab2, tab3 = st.tabs(["Relevance", "User Rating", "Processing Time"])
    
    with tab1:
        fig = px.line(df_daily, x='created_at', y='relevance_score', color='technique',
                     title='Relevance Score Trends',
                     labels={'created_at': 'Date', 'relevance_score': 'Relevance Score'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = px.line(df_daily, x='created_at', y='overall_rating', color='technique',
                     title='User Rating Trends',
                     labels={'created_at': 'Date', 'overall_rating': 'User Rating'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        fig = px.line(df_daily, x='created_at', y='processing_time', color='technique',
                     title='Processing Time Trends',
                     labels={'created_at': 'Date', 'processing_time': 'Processing Time (s)'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def create_feedback_analysis(performance_history: List[Dict]) -> None:
    """Create analysis of user feedback patterns"""
    if not performance_history:
        return
    
    st.subheader("💬 User Feedback Analysis")
    
    df = pd.DataFrame(performance_history)
    
    # Filter rows with feedback
    feedback_df = df[df['overall_rating'].notna()]
    
    if feedback_df.empty:
        st.info("No user feedback collected yet. Encourage users to rate responses!")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Rating distribution
        fig = px.histogram(feedback_df, x='overall_rating', 
                          title='Distribution of User Ratings',
                          nbins=5, range_x=[0.5, 5.5])
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Average ratings by technique
        avg_ratings = feedback_df.groupby('technique')['overall_rating'].mean().reset_index()
        fig = px.bar(avg_ratings, x='technique', y='overall_rating',
                    title='Average Rating by Technique',
                    color='overall_rating', color_continuous_scale='RdYlGn')
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed feedback breakdown
    if 'feedback_comments' in feedback_df.columns:
        st.subheader("📝 Recent Feedback Comments")
        
        # Show recent comments
        recent_feedback = feedback_df[feedback_df['feedback_comments'].notna()].tail(5)
        
        for _, row in recent_feedback.iterrows():
            with st.expander(f"{row['technique']} - Rating: {row['overall_rating']}/5"):
                st.write(f"**Query:** {row['query'][:100]}...")
                st.write(f"**Comment:** {row['feedback_comments']}")
                st.write(f"**Date:** {row['created_at']}")

def create_detailed_metrics_table(comparison_data: Dict[str, Any]) -> None:
    """Create detailed metrics table"""
    if not comparison_data:
        return
    
    st.subheader("📋 Detailed Metrics Table")
      # Convert to DataFrame and format
    df = pd.DataFrame.from_dict(comparison_data, orient='index')
    
    # Replace None values with 0 for numeric columns and format
    numeric_columns = ['avg_relevance', 'avg_faithfulness', 'avg_completeness', 
                      'avg_semantic_similarity', 'avg_processing_time', 'avg_response_length', 'avg_user_rating']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # Round numeric columns
    numeric_columns_in_df = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns_in_df] = df[numeric_columns_in_df].round(3)
    
    # Rename columns for better display
    column_mapping = {
        'total_queries': 'Total Queries',
        'avg_relevance': 'Avg Relevance',
        'avg_faithfulness': 'Avg Faithfulness',
        'avg_completeness': 'Avg Completeness',
        'avg_semantic_similarity': 'Avg Semantic Sim.',
        'avg_processing_time': 'Avg Time (s)',
        'avg_response_length': 'Avg Length',
        'avg_user_rating': 'Avg Rating',
        'feedback_count': 'Feedback Count'
    }
    
    df_display = df.rename(columns=column_mapping)
    
    # Create format dictionary for non-None values only
    format_dict = {}
    if 'Avg Relevance' in df_display.columns:
        format_dict['Avg Relevance'] = lambda x: f'{x:.3f}' if pd.notna(x) and x is not None else 'N/A'
    if 'Avg Faithfulness' in df_display.columns:
        format_dict['Avg Faithfulness'] = lambda x: f'{x:.3f}' if pd.notna(x) and x is not None else 'N/A'
    if 'Avg Completeness' in df_display.columns:
        format_dict['Avg Completeness'] = lambda x: f'{x:.3f}' if pd.notna(x) and x is not None else 'N/A'
    if 'Avg Semantic Sim.' in df_display.columns:
        format_dict['Avg Semantic Sim.'] = lambda x: f'{x:.3f}' if pd.notna(x) and x is not None else 'N/A'
    if 'Avg Time (s)' in df_display.columns:
        format_dict['Avg Time (s)'] = lambda x: f'{x:.3f}' if pd.notna(x) and x is not None else 'N/A'
    if 'Avg Rating' in df_display.columns:
        format_dict['Avg Rating'] = lambda x: f'{x:.1f}' if pd.notna(x) and x is not None else 'N/A'
    
    # Apply formatting only if we have data
    if format_dict:
        styled_df = df_display.style.format(format_dict)
        
        # Apply background gradient only to columns that exist and have numeric data
        gradient_cols = [col for col in ['Avg Relevance', 'Avg Faithfulness', 'Avg Completeness'] 
                        if col in df_display.columns and df_display[col].dtype in ['float64', 'int64']]
        
        if gradient_cols:
            styled_df = styled_df.background_gradient(subset=gradient_cols, cmap='RdYlGn', vmin=0, vmax=1)
    else:
        styled_df = df_display
    
    st.dataframe(styled_df, use_container_width=True)

def create_export_section(evaluation_manager) -> None:
    """Create section for exporting evaluation data"""
    st.subheader("📁 Export & Data Management")
    
    # Export options
    st.write("**📊 Export Options:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Analytics Report"):
            try:
                filename = evaluation_manager.export_evaluation_data()
                with open(filename, 'r') as f:
                    data = f.read()
                
                st.download_button(
                    label="📥 Download Analytics Report",
                    data=data,
                    file_name=filename,
                    mime="application/json"
                )
                st.success(f"Analytics report generated: {filename}")
            except Exception as e:
                st.error(f"Error generating report: {e}")
    
    with col2:
        if st.button("📈 Export Performance CSV"):
            try:
                performance_data = evaluation_manager.get_performance_history(limit=1000)
                if performance_data:
                    df = pd.DataFrame(performance_data)
                    csv = df.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"rag_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    st.success("Performance data exported to CSV")
                else:
                    st.warning("No performance data available to export")
            except Exception as e:
                st.error(f"Error exporting CSV: {e}")
    
    with col3:
        # Export by date range
        if st.button("� Export by Date Range"):
            st.info("🚧 Feature coming soon - export data for specific date ranges")
    
    st.divider()
    
    # Data management options
    st.write("**🗑️ Data Management:**")
    
    # Add the quick clear section
    create_quick_clear_section()
    
    st.divider()
    
    # Database information
    st.write("**📊 Database Information:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Database info
        db_size = get_db_size(evaluation_manager.database.db_path)
        st.metric("Database Size", f"{db_size:.1f} MB")
    
    with col2:
        # Show total records
        try:
            import sqlite3
            conn = sqlite3.connect(evaluation_manager.database.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM evaluations")
            record_count = cursor.fetchone()[0]
            conn.close()
            st.metric("Total Records", record_count)
        except:
            st.metric("Total Records", "N/A")
    
    with col3:
        # Show database location
        st.info(f"""
        **Database Location:**
        `{evaluation_manager.database.db_path}`
        """)
    
    # Advanced options
    with st.expander("⚙️ Advanced Database Options", expanded=False):
        st.warning("⚠️ **Advanced users only** - These operations can affect your data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔧 Optimize Database"):
                try:
                    import sqlite3
                    conn = sqlite3.connect(evaluation_manager.database.db_path)
                    conn.execute("VACUUM")
                    conn.close()
                    st.success("✅ Database optimized successfully")
                except Exception as e:
                    st.error(f"Error optimizing database: {e}")
        
        with col2:
            if st.button("📋 Show Database Schema"):
                try:
                    import sqlite3
                    conn = sqlite3.connect(evaluation_manager.database.db_path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table'")
                    schema = cursor.fetchall()
                    conn.close()
                    
                    st.code("\n\n".join([s[0] for s in schema if s[0]]), language="sql")
                except Exception as e:
                    st.error(f"Error showing schema: {e}")

def get_db_size(db_path: str) -> float:
    """Get database file size in MB"""
    try:
        import os
        return os.path.getsize(db_path) / (1024 * 1024)
    except:
        return 0.0

def display_analytics_dashboard(evaluation_manager):
    """Main function to display the complete analytics dashboard"""
    st.title("📊 RAG Analytics Dashboard")
    
    # Add a prominent clear analytics option at the top
    with st.container():
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col2:
            if st.button("🗑️ Clear All Analytics", type="secondary", help="Reset all evaluation data and start fresh"):
                clear_analytics_data(evaluation_manager)
        
        with col3:
            # Show database status
            try:
                import sqlite3
                conn = sqlite3.connect(evaluation_manager.database.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM evaluations")
                record_count = cursor.fetchone()[0]
                conn.close()
                st.metric("📊 Total Records", record_count, help="Number of evaluation records in database")
            except:
                st.metric("📊 Total Records", "N/A")
    
    st.divider()
    
    # Get data
    comparison_data = evaluation_manager.get_technique_comparison()
    performance_history = evaluation_manager.get_performance_history(limit=500)
    
    # Performance overview
    create_performance_overview(comparison_data)
    
    st.divider()
    
    # Technique comparison
    if comparison_data:
        create_technique_comparison_chart(comparison_data)
        
        st.divider()
        
        # Performance metrics
        create_performance_metrics_chart(comparison_data)
        
        st.divider()
        
        # Time series
        if performance_history:
            create_time_series_chart(performance_history)
            
            st.divider()
        
        # Feedback analysis
        create_feedback_analysis(performance_history)
        
        st.divider()
        
        # Detailed table
        create_detailed_metrics_table(comparison_data)
        
        st.divider()
        
        # Export section
        create_export_section(evaluation_manager)
        
    else:
        st.info("""
        🚀 **Get Started with Analytics**
        
        Start using the chatbot to see detailed analytics here:
        1. Upload some documents
        2. Ask questions using different RAG techniques
        3. Provide feedback on responses
        4. Watch the analytics come to life!
        """)

def clear_analytics_data(evaluation_manager):
    """Enhanced function to clear analytics data with better user experience"""
    # Use a unique key for the confirmation state
    confirm_key = "confirm_clear_analytics"
    
    if not st.session_state.get(confirm_key, False):
        # First click - show confirmation
        st.session_state[confirm_key] = True
        st.warning("⚠️ **Are you sure?** This will permanently delete ALL evaluation data including:")
        st.write("- All query evaluations and responses")
        st.write("- User feedback and ratings") 
        st.write("- Performance metrics and history")
        st.write("- Technique comparison data")
        st.info("💡 **Tip:** Consider exporting your data first using the export options below.")
        
        # Show confirmation buttons
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("✅ Yes, Clear All", type="primary", key="confirm_clear_yes"):
                perform_database_reset(evaluation_manager)
                st.session_state[confirm_key] = False
        
        with col2:
            if st.button("❌ Cancel", key="confirm_clear_no"):
                st.session_state[confirm_key] = False
                st.success("✅ Cancelled - No data was deleted")
                time.sleep(1)
                st.rerun()
        
        st.stop()  # Stop execution to show confirmation dialog
    
    else:
        # Reset confirmation state if we somehow get here
        st.session_state[confirm_key] = False

def perform_database_reset(evaluation_manager):
    """Actually perform the database reset operation using the safe method"""
    try:
        import sqlite3
        
        db_path = evaluation_manager.database.db_path
        
        # Get current record count before deletion
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM evaluations")
        records_before = cursor.fetchone()[0]
        
        if records_before == 0:
            conn.close()
            st.info("ℹ️ Database is already empty - no records to clear")
            return
        
        # Simple and safe approach: just delete all records
        cursor.execute("DELETE FROM evaluations")
        conn.commit()
        
        # Verify the deletion worked
        cursor.execute("SELECT COUNT(*) FROM evaluations")
        records_after = cursor.fetchone()[0]
        
        conn.close()
        
        # Show success message
        st.success(f"✅ **Analytics Cleared Successfully!**")
        st.info(f"📊 Removed {records_before} evaluation records")
        
        if records_after == 0:
            st.success("🔍 **Verification:** Database is now empty")
        else:
            st.warning(f"⚠️ **Warning:** {records_after} records still remain")
        
        st.info("🔄 **Refresh the page** to see the updated analytics dashboard")
        
        # Clear any session state related to evaluations
        if 'pending_feedback' in st.session_state:
            st.session_state.pending_feedback = {}
        
        # Add a manual refresh button
        if st.button("🔄 Refresh Dashboard", type="primary"):
            st.rerun()
        
        # Auto-refresh after a short delay
        time.sleep(1)
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ **Error clearing analytics data:** {str(e)}")
        st.warning("Please try again or contact support if the problem persists.")
        
        # Provide detailed error information for debugging
        with st.expander("🔧 Error Details (for debugging)"):
            st.code(f"Error type: {type(e).__name__}")
            st.code(f"Error message: {str(e)}")
            st.code(f"Database path: {evaluation_manager.database.db_path if evaluation_manager else 'N/A'}")
            
            # Show available tables
            try:
                import sqlite3
                conn = sqlite3.connect(evaluation_manager.database.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                conn.close()
                st.code(f"Available tables: {tables}")
            except:
                st.code("Could not list tables")

def create_quick_clear_section():
    """Create a quick clear section for the sidebar or main area"""
    with st.expander("🗑️ Clear Analytics Data", expanded=False):
        st.write("**Quick Actions:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Clear last session only
            if st.button("Clear Last Session", help="Remove data from the current session only"):
                st.info("🚧 Feature coming soon - will clear only current session data")
        
        with col2:
            # Clear by technique
            techniques = ["Adaptive RAG", "CRAG", "Document Augmentation", "Basic RAG", "Explainable Retrieval"]
            selected_technique = st.selectbox("Clear by Technique:", ["Select..."] + techniques)
            
            if st.button("Clear Technique Data", disabled=(selected_technique == "Select...")):
                if selected_technique != "Select...":
                    clear_technique_data(selected_technique)

def clear_technique_data(technique: str):
    """Clear data for a specific RAG technique"""
    try:
        import sqlite3
        from src.evaluation_framework import EvaluationManager
        
        # Get evaluation manager (assuming it's available in session state or create new)
        evaluation_manager = EvaluationManager()
        
        conn = sqlite3.connect(evaluation_manager.database.db_path)
        cursor = conn.cursor()
        
        # Count records before deletion
        cursor.execute("SELECT COUNT(*) FROM evaluations WHERE technique = ?", (technique,))
        records_before = cursor.fetchone()[0]
        
        if records_before > 0:
            # Delete records for the specific technique
            cursor.execute("DELETE FROM evaluations WHERE technique = ?", (technique,))
            conn.commit()
            
            st.success(f"✅ Cleared {records_before} records for **{technique}**")
            st.info("🔄 Refresh the page to see updated analytics")
        else:
            st.warning(f"ℹ️ No data found for **{technique}**")
        
        conn.close()
        
    except Exception as e:
        st.error(f"❌ Error clearing {technique} data: {str(e)}")

def safe_clear_database(evaluation_manager):
    """Safely clear database without dealing with auto-increment sequences"""
    try:
        import sqlite3
        import os
        
        db_path = evaluation_manager.database.db_path
        
        # Get current record count before deletion
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM evaluations")
        records_before = cursor.fetchone()[0]
        conn.close()
        
        if records_before == 0:
            st.info("ℹ️ Database is already empty - no records to clear")
            return
        
        # Method 1: Simple DELETE (recommended)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM evaluations")
        conn.commit()
        conn.close()
        
        st.success(f"✅ **Analytics Cleared Successfully!**")
        st.info(f"📊 Removed {records_before} evaluation records")
        
        # Verify the deletion worked
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM evaluations")
        records_after = cursor.fetchone()[0]
        conn.close()
        
        if records_after == 0:
            st.success("🔍 **Verification:** Database is now empty")
        else:
            st.warning(f"⚠️ **Warning:** {records_after} records still remain")
        
        return True
        
    except Exception as e:
        st.error(f"❌ **Error clearing analytics data:** {str(e)}")
        
        # Provide detailed error information for debugging
        with st.expander("🔧 Error Details (for debugging)"):
            st.code(f"Error type: {type(e).__name__}")
            st.code(f"Error message: {str(e)}")
            st.code(f"Database path: {evaluation_manager.database.db_path}")
        
        return False

def recreate_database(evaluation_manager):
    """Alternative method: recreate the database from scratch"""
    try:
        import sqlite3
        import os
        
        db_path = evaluation_manager.database.db_path
        
        # Get current record count
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM evaluations")
            records_before = cursor.fetchone()[0]
            conn.close()
        else:
            records_before = 0
        
        # Close any existing connections
        if hasattr(evaluation_manager.database, '_connection'):
            evaluation_manager.database._connection.close()
        
        # Remove the old database file
        if os.path.exists(db_path):
            os.remove(db_path)
            st.info(f"🗑️ Removed old database with {records_before} records")
        
        # Recreate the database with fresh schema
        evaluation_manager.database._init_database()
        
        st.success("✅ **Database recreated successfully!**")
        st.info("📊 Fresh database created with clean schema")
        
        return True
        
    except Exception as e:
        st.error(f"❌ **Error recreating database:** {str(e)}")
        return False
