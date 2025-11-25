"""
Initialize PostgreSQL schema by running init_postgres.sql
This script creates all necessary tables in the fraud_detection database.
"""

import psycopg2
import yaml
import os
import sys
import re
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def load_config():
    """Load configuration from config.yaml"""
    config_path = project_root / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def parse_sql_statements(sql_script):
    """
    Parse SQL script into individual statements, handling:
    - Dollar-quoted strings ($$ ... $$)
    - Multi-line statements
    - Comments
    """
    statements = []
    current_statement = []
    in_dollar_quote = False
    dollar_quote_tag = None
    in_single_quote = False
    in_double_quote = False
    
    lines = sql_script.split('\n')
    
    for line in lines:
        # Remove inline comments (-- comments)
        if '--' in line and not in_single_quote and not in_double_quote and not in_dollar_quote:
            comment_pos = line.find('--')
            line = line[:comment_pos]
        
        # Skip comment-only lines
        stripped = line.strip()
        if stripped.startswith('--') or not stripped:
            continue
        
        # Track quotes and dollar quotes
        i = 0
        while i < len(line):
            char = line[i]
            
            # Handle dollar-quoted strings ($$ ... $$)
            if not in_single_quote and not in_double_quote:
                if line[i:i+2] == '$$':
                    if not in_dollar_quote:
                        # Find the tag (could be $tag$)
                        j = i + 2
                        tag_end = line.find('$', j)
                        if tag_end != -1:
                            dollar_quote_tag = line[i:tag_end+1]
                            in_dollar_quote = True
                            i = tag_end
                    else:
                        # Check if this closes the dollar quote
                        if line[i:i+len(dollar_quote_tag)] == dollar_quote_tag:
                            in_dollar_quote = False
                            dollar_quote_tag = None
                            i += len(dollar_quote_tag) - 1
            
            # Handle single quotes (only if not in dollar quote)
            if not in_dollar_quote:
                if char == "'" and not in_double_quote:
                    # Check for escaped quotes
                    if i + 1 < len(line) and line[i+1] == "'":
                        i += 2
                        continue
                    in_single_quote = not in_single_quote
                elif char == '"' and not in_single_quote:
                    in_double_quote = not in_double_quote
            
            i += 1
        
        current_statement.append(line)
        
        # Check if line ends a statement (semicolon outside quotes)
        if ';' in line and not in_single_quote and not in_double_quote and not in_dollar_quote:
            # Split by semicolon
            parts = line.split(';')
            if len(parts) > 1:
                # Add the part before semicolon to current statement
                current_statement[-1] = parts[0]
                statement = '\n'.join(current_statement).strip()
                if statement:
                    statements.append(statement)
                # Start new statement with part after semicolon
                current_statement = [parts[1]] if parts[1].strip() else []
            else:
                statement = '\n'.join(current_statement).strip()
                if statement:
                    statements.append(statement)
                current_statement = []
    
    # Add any remaining statement
    if current_statement:
        statement = '\n'.join(current_statement).strip()
        if statement:
            statements.append(statement)
    
    return statements

def init_schema():
    """Initialize PostgreSQL schema"""
    config = load_config()
    pg_config = config['database']['postgres']
    
    # Read SQL file
    sql_file = project_root / "scripts" / "init_postgres.sql"
    with open(sql_file, 'r') as f:
        sql_script = f.read()
    
    # Connect to PostgreSQL
    try:
        conn = psycopg2.connect(
            host=pg_config['host'],
            port=pg_config['port'],
            database=pg_config['database'],
            user=pg_config['user'],
            password=pg_config['password']
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        print(f"✅ Connected to PostgreSQL at {pg_config['host']}:{pg_config['port']}")
        print(f"📊 Database: {pg_config['database']}")
        print("\n" + "="*60)
        print("Creating database schema...")
        print("="*60 + "\n")
        
        # Parse SQL statements properly
        statements = parse_sql_statements(sql_script)
        
        executed_count = 0
        for i, statement in enumerate(statements, 1):
            statement = statement.strip()
            if not statement:
                continue
            
            # Skip SELECT statements for now (we'll run them at the end)
            if statement.upper().strip().startswith('SELECT'):
                continue
            
            try:
                cursor.execute(statement)
                # Get the statement type for logging
                stmt_type = statement.split()[0].upper() if statement.split() else "UNKNOWN"
                if stmt_type in ['CREATE', 'INSERT', 'DROP', 'ALTER', 'GRANT']:
                    print(f"✅ Executed: {stmt_type} statement")
                    executed_count += 1
            except psycopg2.errors.DuplicateTable:
                stmt_type = statement.split()[0].upper() if statement.split() else "UNKNOWN"
                print(f"ℹ️  Skipped (already exists): {stmt_type}")
            except psycopg2.errors.DuplicateObject:
                stmt_type = statement.split()[0].upper() if statement.split() else "UNKNOWN"
                print(f"ℹ️  Skipped (already exists): {stmt_type}")
            except Exception as e:
                # Check if it's a "relation does not exist" error for DROP statements
                if "does not exist" in str(e) and statement.upper().strip().startswith('DROP'):
                    print(f"ℹ️  Skipped (does not exist): DROP statement")
                else:
                    print(f"⚠️  Warning: {str(e)[:100]}")
                    # Print the problematic statement for debugging
                    if "relation" in str(e).lower():
                        print(f"   Statement: {statement[:200]}...")
        
        print(f"\n✅ Executed {executed_count} statements")
        print("\n" + "="*60)
        print("✅ Schema initialization complete!")
        print("="*60)
        
        # Verify tables were created
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            ORDER BY table_name;
        """)
        tables = cursor.fetchall()
        
        print(f"\n📋 Created tables ({len(tables)}):")
        for table in tables:
            # Count rows in each table
            cursor.execute(f"SELECT COUNT(*) FROM {table[0]};")
            count = cursor.fetchone()[0]
            print(f"   • {table[0]}: {count} rows")
        
        # Run the final SELECT statement to show summary
        final_select = """
        SELECT 
            'Database initialized successfully!' as status,
            (SELECT COUNT(*) FROM users) as users_count,
            (SELECT COUNT(*) FROM locations) as locations_count,
            (SELECT COUNT(*) FROM transactions) as transactions_count,
            (SELECT COUNT(*) FROM transactions WHERE is_fraud = TRUE) as fraud_count;
        """
        try:
            cursor.execute(final_select)
            results = cursor.fetchall()
            if results:
                columns = [desc[0] for desc in cursor.description]
                print(f"\n📊 Summary:")
                print("-" * 60)
                for col in columns:
                    print(f"{col:30}", end="")
                print()
                print("-" * 60)
                for row in results:
                    for val in row:
                        print(f"{str(val):30}", end="")
                    print()
                print("-" * 60)
        except Exception as e:
            print(f"⚠️  Could not run summary query: {e}")
        
        cursor.close()
        conn.close()
        
        print("\n✅ PostgreSQL schema initialized successfully!")
        return True
        
    except psycopg2.Error as e:
        print(f"❌ PostgreSQL error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = init_schema()
    sys.exit(0 if success else 1)

