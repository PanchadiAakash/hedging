import psycopg2
import config
import pandas as pd
import requests

# --- Metabase data ---

def get_question_data(question):
    """Accessing Metabase API through Python and extracting data from saved
    questions to pandas dataframe
    """

    resp = requests.post(
        "https://zata.uat.zetwerk.com/api/session",
        headers={"Content-Type": "application/json"},
        json={
            "username": config.METABASE_USERNAME,
            "password": config.METABASE_PASSWORD,
        },
    )
    token = resp.json()["id"]

    request_url = "https://zata.uat.zetwerk.com/api/card/" + str(question) + "/query/json"

    res = requests.post(
        request_url,
        headers={"Content-Type": "application/json", "X-Metabase-Session": token},
    )
    try:
        return pd.DataFrame(res.json())
    except Exception as e:
        with open("logs/response_{}.txt".format(question), "w") as f:
            f.write(res.text)
        raise e


def pg_connect():
    conn = psycopg2.connect(**config.POSTGRES_DB)
    return conn

def create_paper_sell_entry_push(df):
    conn = pg_connect()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS Paper_sell_entry (
                    position_id VARCHAR PRIMARY KEY,
                    entry_id VARCHAR,
                    product VARCHAR,
                    status VARCHAR,
                    artefact_status VARCHAR,
                    supplier_name VARCHAR,
                    supplier_group_name VARCHAR,
                    num_lots NUMERIC(20,2),
                    lot_size NUMERIC(20,2),
                    total_quantity NUMERIC(20,2),
                    paper_open_quantity NUMERIC(20,2),
                    bill_open_quantity NUMERIC(20,2),
                    price_rate NUMERIC(20,2),
                    expiry_date DATE,
                    transaction_date DATE,
                    exchange VARCHAR,
                    currency VARCHAR,
                    conversion_rate NUMERIC(20,2),
                    roll_over_quantity NUMERIC(20,2) DEFAULT 0,
                    createdat TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            insert_query = """
                INSERT INTO Paper_sell_entry (
                    position_id, entry_id, product, status,artefact_status, supplier_name, supplier_group_name,
                    num_lots, lot_size, total_quantity, paper_open_quantity, bill_open_quantity, price_rate, expiry_date,
                    transaction_date, exchange, currency, conversion_rate, roll_over_quantity
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (position_id) DO NOTHING
            """

            data = [
                (
                    str(row["position_id"]),
        str(row["entry_id"]),
        str(row["product"]),
        str(row["status"]),
        str('OPEN'),
        str(row["supplier_name"]),
        str(row["supplier_group_name"]),
        float(row["num_lots"]) if pd.notnull(row["num_lots"]) else None,
        float(row["lot_size"]) if pd.notnull(row["lot_size"]) else None,
        float(row["total_quantity"]) if pd.notnull(row["total_quantity"]) else None,
        float(row["paper_open_quantity"]) if pd.notnull(row["paper_open_quantity"]) else None,
        float(row["bill_open_quantity"]) if pd.notnull(row["bill_open_quantity"]) else None,
        float(row["price_rate"]) if pd.notnull(row["price_rate"]) else None,
        row.expiry_date,
        row.transaction_date,
        str(row["exchange"]),
        str(row["currency"]),
        float(row["conversion_rate"]) if pd.notnull(row["conversion_rate"]) else None,
        float(row["roll_over_quantity"]) if pd.notnull(row["roll_over_quantity"]) else None,
                )
                for _, row in df.iterrows()
            ]
            cur.executemany(insert_query, data)

        conn.commit()
    finally:
        conn.close()

def create_paper_buy_entry_push(df):
    conn = pg_connect()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS Paper_buy_entry (
                    position_id VARCHAR PRIMARY KEY,
                    entry_id VARCHAR,
                    product VARCHAR,
                    status VARCHAR,
                    artefact_status VARCHAR,
                    customer_name VARCHAR,
                    customer_group_name VARCHAR,
                    num_lots NUMERIC(20,2),
                    lot_size NUMERIC(20,2),
                    total_quantity NUMERIC(20,6),
                    paper_open_quantity NUMERIC(20,6),
                    invoice_open_quantity NUMERIC(20,6),                        
                    price_rate NUMERIC(20,2),
                    expiry_date DATE,
                    transaction_date DATE,
                    exchange VARCHAR,
                    currency VARCHAR,
                    conversion_rate NUMERIC(20,2),
                    roll_over_quantity NUMERIC(20,2) DEFAULT 0,
                    createdat TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            insert_query = """
                INSERT INTO Paper_buy_entry (
                    position_id, entry_id, product, status, artefact_status, customer_name, customer_group_name,
                    num_lots, lot_size, total_quantity, paper_open_quantity, invoice_open_quantity, price_rate, expiry_date,
                    transaction_date, exchange, currency, conversion_rate, roll_over_quantity
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (position_id) DO NOTHING
            """

            data = [
                (
                    str(row["position_id"]),
        str(row["entry_id"]),
        str(row["product"]),
        str(row["status"]),
        str('OPEN'),
        str(row["customer_name"]),
        str(row["customer_group_name"]),
        float(row["num_lots"]) if pd.notnull(row["num_lots"]) else None,
        float(row["lot_size"]) if pd.notnull(row["lot_size"]) else None,
        float(row["total_quantity"]) if pd.notnull(row["total_quantity"]) else None,
        float(row["paper_open_quantity"]) if pd.notnull(row["paper_open_quantity"]) else None,
        float(row["invoice_open_quantity"]) if pd.notnull(row["invoice_open_quantity"]) else None,
        float(row["price_rate"]) if pd.notnull(row["price_rate"]) else None,
        row.expiry_date,
        row.transaction_date,
        str(row["exchange"]),
        str(row["currency"]),
        float(row["conversion_rate"]) if pd.notnull(row["conversion_rate"]) else None,
        float(row["roll_over_quantity"]) if pd.notnull(row["roll_over_quantity"]) else None,
                )
                for _, row in df.iterrows()
            ]
            cur.executemany(insert_query, data)

        conn.commit()
    finally:
        conn.close()

def drop_data_paper():
    conn = pg_connect()
    try:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE Paper_sell_entry")
            cur.execute("TRUNCATE TABLE Paper_buy_entry")
            cur.execute("TRUNCATE TABLE hedge_paper_settle_entry")
        conn.commit()
    finally:
        conn.close()

def get_table_data(table_name):
    conn = pg_connect()
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT * FROM {table_name}")
            rows = cur.fetchall()
            colnames = [desc[0] for desc in cur.description]
            df = pd.DataFrame(rows, columns=colnames)
            return df
    finally:
        conn.close()

def update_table_buy_completly(open_quantity, position_id):
    conn = pg_connect()
    try:
        cur = conn.cursor()
        if open_quantity == 0:
            cur.execute("Update Paper_buy_entry set paper_open_quantity = 0, status = 'CLOSED' where position_id = %s", (position_id,))
        elif open_quantity < 0:
            print("Open quantity cannot be negative")
        else:
            cur.execute("Update Paper_buy_entry set paper_open_quantity = %s where position_id = %s", (open_quantity, position_id))
        conn.commit()
    except Exception as e:
        print(f"Error updating Paper_buy_entry: {e}")
    finally:
        conn.close()

def update_table_sell_completly(open_quantity, position_id):
    conn = pg_connect()
    try:
        cur = conn.cursor()
        if open_quantity == 0:
            cur.execute("Update Paper_sell_entry set paper_open_quantity = 0, status = 'CLOSED' where position_id = %s", (position_id,))
        elif open_quantity < 0:
            print("Open quantity cannot be negative")
        else:
            cur.execute("Update Paper_sell_entry set paper_open_quantity = %s where position_id = %s", (open_quantity, position_id))
        conn.commit()
    except Exception as e:
        print(f"Error updating Paper_sell_entry: {e}")
    finally:
        conn.close()

def create_settlement_entry_push(df):
    conn = pg_connect()
    try:
        with conn.cursor() as cur:
            # Create table if it doesn't exist
            cur.execute("""
                CREATE TABLE IF NOT EXISTS hedge_paper_settle_entry (
                    buy_entry_id VARCHAR,
                    sell_entry_id VARCHAR,
                    buy_position_id VARCHAR,
                    sell_position_id VARCHAR,
                    hedge_quantity NUMERIC(20,2),
                    product VARCHAR,
                    buy_price_rate NUMERIC(20,2),
                    sell_price_rate NUMERIC(20,2),
                    buy_exchange VARCHAR,
                    sell_exchange VARCHAR,
                    settlement_date DATE,
                    createdat TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Prepare insert statement
            insert_query = """
                INSERT INTO hedge_paper_settle_entry (
                    buy_entry_id, sell_entry_id, buy_position_id, sell_position_id,
                    hedge_quantity, product, buy_price_rate, sell_price_rate,
                    buy_exchange, sell_exchange, settlement_date
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """

            # Prepare data
            data = [
                (
                    str(row["buy_entry_id"]),
                    str(row["sell_entry_id"]),
                    str(row["buy_position_id"]),
                    str(row["sell_position_id"]),
                    float(row["hedge_quantity"]) if pd.notnull(row["hedge_quantity"]) else None,
                    str(row["product"]),
                    float(row["buy_price"]) if pd.notnull(row["buy_price"]) else None,
                    float(row["sell_price"]) if pd.notnull(row["sell_price"]) else None,
                    str(row["buy_exchange"]),
                    str(row["sell_exchange"]),
                    row["settlement_date"] if "settlement_date" in row else None
                )
                for _, row in df.iterrows()
            ]

            # Execute inserts safely inside the same context
            cur.executemany(insert_query, data)

            # ✅ Commit while cursor is still open
            conn.commit()

        print("✅ Settlement entries successfully inserted.")
    except Exception as e:
        conn.rollback()
        print(f"❌ Error inserting settlement entries: {e}")
    finally:
        conn.close()



def update_paper_sell_bill(open_quantity, position_id):
    conn = pg_connect()
    try:
        cur = conn.cursor()
        if open_quantity == 0:
            cur.exceute("UPDATE paper_sell_entry SET bill_open_quantity = 0 , artefact_status = 'CLOSED' WHERE position_id = %s",(position_id,))
        elif open_quantity < 0:
            print("Open quantity cannot be negative")
        else:
            cur.exceute("UPDATE paper_sell_entry SET bill_open_quantity = %s WHERE position_id = %s", (open_quantity, position_id))
        conn.commit()
    finally:
        conn.close()

def create_paper_invoice_tagging(df):
    conn = pg_connect()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS paper_invoice_tagging (
                    position_id VARCHAR,
                    invoicenumber VARCHAR,
                    product VARCHAR,
                    tagged_qty NUMERIC(20,2),
                    paper_remaining_after NUMERIC(20,2),
                    invoice_remaining_after NUMERIC(20,2),
                    createdat TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            insert_query = """
            INSERT INTO paper_invoice_tagging (
                position_id, invoicenumber, product, tagged_qty, paper_remaining_after, invoice_remaining_after
            ) VALUES (%s,%s,%s,%s,%s,%s)
            """
            data = [
            (
                str(row["position_id"]),
                str(row["invoicenumber"]),
                str(row["product"]),
                float(row["tagged_qty"]) if pd.notnull(row["tagged_qty"]) else None,
                float(row["paper_remaining_after"]) if pd.notnull(row["paper_remaining_after"]) else None,
                float(row["invoice_remaining_after"]) if pd.notnull(row["invoice_remaining_after"]) else None, 
            )
             for _, row in df.iterrows()
            ]
            cur.executemany(insert_query, data)
            conn.commit()
    finally:
        conn.close()


def update_paper_buy_invoice_df(df):
    conn = pg_connect()
    try:
        cur = conn.cursor()
        
        for _, row in df.iterrows():
            open_quantity = row['remaining_qty']
            position_id = row['position_id']

            if open_quantity < 0:
                print(f"⚠️ Skipped position_id {position_id}: Open quantity cannot be negative")
            elif open_quantity == 0:
                cur.execute("""
                    UPDATE paper_buy_entry
                    SET invoice_open_quantity = 0,
                        artefact_status = 'CLOSED'
                    WHERE position_id = %s
                """, (position_id,))
            else:
                cur.execute("""
                    UPDATE paper_buy_entry
                    SET invoice_open_quantity = %s
                    WHERE position_id = %s
                """, (open_quantity, position_id))

        conn.commit()
        print("✅ Database updated successfully.")
    except Exception as e:
        conn.rollback()
        print(f"❌ Error: {e}")
    finally:
        conn.close()


def update_paper_sell_bill_df(df, open_qty_col, position_id_col):
    """
    Updates 'paper_sell_entry' table based on a DataFrame input.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing sell positions.
        open_qty_col (str): Column name in df for open_quantity.
        position_id_col (str): Column name in df for position_id.
    """
    conn = pg_connect()
    try:
        cur = conn.cursor()

        for _, row in df.iterrows():
            open_quantity = row[open_qty_col]
            position_id = row[position_id_col]

            if open_quantity < 0:
                print(f"⚠️ Skipped position_id {position_id}: Open quantity cannot be negative")
                continue

            if open_quantity == 0:
                cur.execute("""
                    UPDATE paper_sell_entry
                    SET bill_open_quantity = 0,
                        artefact_status = 'CLOSED'
                    WHERE position_id = %s
                """, (position_id,))
            else:
                cur.execute("""
                    UPDATE paper_sell_entry
                    SET bill_open_quantity = %s
                    WHERE position_id = %s
                """, (open_quantity, position_id))

        conn.commit()
        print("✅ paper_sell_entry table updated successfully.")
    except Exception as e:
        conn.rollback()
        print(f"❌ Error during update: {e}")
    finally:
        conn.close()


def create_paper_bill_tagging(df):
    conn = pg_connect()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS paper_bill_tagging (
                    position_id VARCHAR,
                    billnumber VARCHAR,
                    product VARCHAR,
                    tagged_qty NUMERIC(20,2),
                    paper_remaining_after NUMERIC(20,2),
                    bill_remaining_after NUMERIC(20,2),
                    createdat TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            insert_query = """
            INSERT INTO paper_bill_tagging (
                position_id, billnumber, product, tagged_qty, paper_remaining_after, bill_remaining_after
            ) VALUES (%s,%s,%s,%s,%s,%s)
            """
            data = [
            (
                str(row["position_id"]),
                str(row["billnumber"]),
                str(row["product"]),
                float(row["tagged_qty"]) if pd.notnull(row["tagged_qty"]) else None,
                float(row["paper_remaining_after"]) if pd.notnull(row["paper_remaining_after"]) else None,
                float(row["bill_remaining_after"]) if pd.notnull(row["bill_remaining_after"]) else None, 
            )
             for _, row in df.iterrows()
            ]
            cur.executemany(insert_query, data)
            conn.commit()
    finally:
        conn.close()
    
def update_table_buy_roll_over_completly(open_quantity, roll_qty, position_id):
    conn = pg_connect()
    try:
        cur = conn.cursor()
        if open_quantity == 0:
            cur.execute("Update Paper_buy_entry set paper_open_quantity = 0, roll_over_quantity = %s, status = 'CLOSED' where position_id = %s", (roll_qty, position_id))
        elif open_quantity < 0:
            print("Open quantity cannot be negative")
        else:
            cur.execute("Update Paper_buy_entry set paper_open_quantity = %s, roll_over_quantity = %s where position_id = %s", (open_quantity, roll_qty, position_id))
        conn.commit()
    except Exception as e:
        print(f"Error updating Paper_buy_entry: {e}")
    finally:
        conn.close()

    
def update_table_sell_roll_over_completly(open_quantity, roll_qty, position_id):
    conn = pg_connect()
    try:
        cur = conn.cursor()
        if open_quantity == 0:
            cur.execute("Update Paper_sell_entry set paper_open_quantity = 0, roll_over_quantity = %s, status = 'CLOSED' where position_id = %s", (roll_qty, position_id))
        elif open_quantity < 0:
            print("Open quantity cannot be negative")
        else:
            cur.execute("Update Paper_sell_entry set paper_open_quantity = %s, roll_over_quantity = %s where position_id = %s", (open_quantity, roll_qty, position_id))
        conn.commit()
    except Exception as e:
        print(f"Error updating Paper_sell_entry: {e}")
    finally:
        conn.close()


def create_paper_roll_over_tagging(df):
    conn = pg_connect()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS paper_roll_over_tagging (
                    old_entry_id VARCHAR,
                    old_position_id VARCHAR,
                    new_entry_id VARCHAR,
                    buy_position_id VARCHAR,
                    sell_position_id VARCHAR,
                    roll_over_quantity NUMERIC(20,2),
                    product VARCHAR,
                    price_rate NUMERIC(20,2),
                    type VARCHAR,
                    createdat TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            insert_query = """
            INSERT INTO paper_roll_over_tagging (
                old_entry_id, old_position_id, new_entry_id, buy_position_id, sell_position_id,
                roll_over_quantity, product, price_rate, type
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """
            data = [
            (
                str(row["old_entry_id"]),
                str(row["old_position_id"]),
                str(row["new_entry_id"]),
                str(row["buy_position_id"]),
                str(row["sell_position_id"]),
                float(row["roll_over_quantity"]) if pd.notnull(row["roll_over_quantity"]) else None,
                str(row["product"]),
                float(row["price_rate"]) if pd.notnull(row["price_rate"]) else None,
                str(row["type"]),
            )
             for _, row in df.iterrows()
            ]
            cur.executemany(insert_query, data)
            conn.commit()
    finally:
        conn.close()

# if __name__ == "__main__":
#     conn = pg_connect()
#     cur = conn.cursor()
#     cur.execute("ALTER TABLE paper_buy_entry ADD COLUMN roll_over_quantity NUMERIC(20,2) DEFAULT 0;")
#     conn.commit()
#     conn.close()