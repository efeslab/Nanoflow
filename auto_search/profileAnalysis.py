import sqlite3

def getGemvTime(profile_data_path, batch_size, seq_len):
    conn = sqlite3.connect(f'{profile_data_path}/DecAttn.db')
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
        SELECT *
        FROM batched_cuda
        WHERE batch_size = ? AND seq_len = ?
    ORDER BY average_time_ms ASC
        LIMIT 1
    """, (batch_size, seq_len))
    row = cur.fetchone()
    duration = row['average_time_ms']
    print("Fastest CUDA run:", tuple(row), "duration:", duration)
    cur.close()
    conn.close()

    return duration

def getByBatchsize(profile_data_path, name, batch_size):
    conn = sqlite3.connect(f'{profile_data_path}/{name}.db')
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Step 1: Get all table names
    cur.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' 
        AND name NOT LIKE 'sqlite_%';
    """)
    tables = [row["name"] for row in cur.fetchall()]

    best_row = None
    best_duration = float('inf')
    best_table = None

    # Step 2: Iterate over each table
    for table in tables:
        try:
            cur.execute(f"""
                SELECT * FROM {table}
                WHERE batch_size = ?
                ORDER BY average_time_ms ASC
                LIMIT 1
            """, (batch_size,))
            row = cur.fetchone()
            if row and row['average_time_ms'] < best_duration:
                best_row = row
                best_duration = row['average_time_ms']
                best_table = table
        except Exception as e:
            print(f"Skipped table {table} due to error: {e}")

    if best_row:
        print("Fastest run found in table:", best_table)
        print("Row:", tuple(best_row), "Duration:", best_duration)
    else:
        print("No matching entry found for batch size", batch_size)

    cur.close()
    conn.close()
    
    return best_duration