#!/usr/bin/env python3
"""
Loan Data Database Seeder Script
Creates and seeds the loan_data table in SQLite database.
"""

import sqlite3
import os
from pathlib import Path
from datetime import datetime, timedelta
import random
from typing import List, Tuple

# Database configuration
DATABASE_DIR = Path(__file__).parent.parent / "database" / "loan_data"
DATABASE_PATH = DATABASE_DIR / "loan_data.db"

class LoanDataSeeder:
    """Seeder for loan data table."""
    
    def __init__(self):
        """Initialize the seeder."""
        self.db_path = DATABASE_PATH
        self.ensure_directory_exists()
    
    def ensure_directory_exists(self):
        """Ensure the database directory exists."""
        DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    
    def create_connection(self) -> sqlite3.Connection:
        """Create a database connection."""
        try:
            conn = sqlite3.connect(self.db_path)
            print(f"✅ Connected to SQLite database: {self.db_path}")
            return conn
        except sqlite3.Error as e:
            print(f"❌ Error connecting to database: {e}")
            raise
    
    def create_loan_data_table(self, conn: sqlite3.Connection) -> None:
        """Create the loan_data table."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS loan_data (
            loan_id TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL,
            loan_amount DECIMAL(15,2) NOT NULL,
            interest_rate DECIMAL(5,2) NOT NULL,
            start_date DATE NOT NULL,
            tenure_months INTEGER NOT NULL,
            monthly_emi DECIMAL(10,2) NOT NULL,
            amount_paid DECIMAL(15,2) DEFAULT 0.00,
            next_due_date DATE NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('ACTIVE', 'COMPLETED', 'OVERDUE', 'CLOSED')),
            topup_eligibility BOOLEAN DEFAULT 0,
            prepayment_limit DECIMAL(15,2) DEFAULT 0.00,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        try:
            conn.execute(create_table_sql)
            conn.commit()
            print("✅ Loan data table created successfully")
        except sqlite3.Error as e:
            print(f"❌ Error creating table: {e}")
            raise
    
    def generate_sample_data(self, num_records: int = 50) -> List[Tuple]:
        """Generate sample loan data."""
        sample_data = []
        loan_types = ['HL', 'PL', 'CL', 'AL', 'BL']  # Home, Personal, Car, Auto, Business
        statuses = ['ACTIVE', 'COMPLETED', 'OVERDUE', 'CLOSED']
        status_weights = [0.6, 0.2, 0.1, 0.1]  # Most loans are active
        
        for i in range(1, num_records + 1):
            # Generate loan ID
            loan_type = random.choice(loan_types)
            loan_id = f"{loan_type}{str(i).zfill(6)}"
            
            # Generate customer ID
            customer_id = f"CUST{str(random.randint(10000, 10010))}"
            
            # Generate loan amount (1 lakh to 1 crore)
            loan_amount = round(random.uniform(100000, 10000000), 2)
            
            # Generate interest rate (8% to 18%)    
            interest_rate = round(random.uniform(8.0, 18.0), 2)
            
            # Generate start date (last 5 years)
            start_date = datetime.now() - timedelta(days=random.randint(0, 1825))
            start_date_str = start_date.strftime('%Y-%m-%d')
            
            # Generate tenure (12 to 360 months)
            tenure_months = random.choice([12, 24, 36, 48, 60, 84, 120, 180, 240, 360])
            
            # Calculate EMI (simplified calculation)
            monthly_rate = interest_rate / (12 * 100)
            if monthly_rate > 0:
                emi = loan_amount * (monthly_rate * (1 + monthly_rate)**tenure_months) / ((1 + monthly_rate)**tenure_months - 1)
            else:
                emi = loan_amount / tenure_months
            monthly_emi = round(emi, 2)
            
            # Calculate months elapsed and amount paid
            months_elapsed = min(
                int((datetime.now() - start_date).days / 30),
                tenure_months
            )
            amount_paid = round(monthly_emi * months_elapsed, 2)
            
            # Calculate next due date
            next_due_date = start_date + timedelta(days=(months_elapsed + 1) * 30)
            next_due_date_str = next_due_date.strftime('%Y-%m-%d')
            
            # Determine status
            if months_elapsed >= tenure_months:
                status = 'COMPLETED'
            elif next_due_date < datetime.now() - timedelta(days=30):
                status = 'OVERDUE'
            elif random.random() < 0.05:  # 5% chance of closed
                status = 'CLOSED'
            else:
                status = 'ACTIVE'
            
            # Top-up eligibility (50% chance if active and > 12 months)
            topup_eligibility = (
                status == 'ACTIVE' and 
                months_elapsed > 12 and 
                random.random() < 0.5
            )
            
            # Prepayment limit (10-50% of remaining amount for active loans)
            remaining_amount = max(0, loan_amount - amount_paid)
            if status == 'ACTIVE' and remaining_amount > 0:
                prepayment_limit = round(remaining_amount * random.uniform(0.1, 0.5), 2)
            else:
                prepayment_limit = 0.00
            
            sample_data.append((
                loan_id,
                customer_id,
                loan_amount,
                interest_rate,
                start_date_str,
                tenure_months,
                monthly_emi,
                amount_paid,
                next_due_date_str,
                status,
                topup_eligibility,
                prepayment_limit
            ))
        
        return sample_data
    
    def insert_sample_data(self, conn: sqlite3.Connection, data: List[Tuple]) -> None:
        """Insert sample data into the loan_data table."""
        insert_sql = """
        INSERT INTO loan_data (
            loan_id, customer_id, loan_amount, interest_rate, start_date,
            tenure_months, monthly_emi, amount_paid, next_due_date,
            status, topup_eligibility, prepayment_limit
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        try:
            conn.executemany(insert_sql, data)
            conn.commit()
            print(f"✅ Inserted {len(data)} loan records successfully")
        except sqlite3.Error as e:
            print(f"❌ Error inserting data: {e}")
            raise
    
    def create_indexes(self, conn: sqlite3.Connection) -> None:
        """Create indexes for better query performance."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_customer_id ON loan_data(customer_id);",
            "CREATE INDEX IF NOT EXISTS idx_status ON loan_data(status);",
            "CREATE INDEX IF NOT EXISTS idx_next_due_date ON loan_data(next_due_date);",
            "CREATE INDEX IF NOT EXISTS idx_topup_eligibility ON loan_data(topup_eligibility);"
        ]
        
        try:
            for index_sql in indexes:
                conn.execute(index_sql)
            conn.commit()
            print("✅ Database indexes created successfully")
        except sqlite3.Error as e:
            print(f"❌ Error creating indexes: {e}")
            raise
    
    def display_sample_records(self, conn: sqlite3.Connection, limit: int = 5) -> None:
        """Display sample records from the table."""
        try:
            cursor = conn.execute(f"SELECT * FROM loan_data LIMIT {limit}")
            records = cursor.fetchall()
            
            print(f"\n📊 Sample Records (showing {len(records)} out of total):")
            print("-" * 120)
            print(f"{'Loan ID':<10} {'Customer':<12} {'Amount':<12} {'Rate':<6} {'EMI':<10} {'Status':<10} {'Top-up':<8}")
            print("-" * 120)
            
            for record in records:
                loan_id, customer_id, amount, rate, _, _, emi, _, _, status, topup, _ = record[:12]
                print(f"{loan_id:<10} {customer_id:<12} {amount:<12,.0f} {rate:<6}% {emi:<10,.0f} {status:<10} {'Yes' if topup else 'No':<8}")
            
            # Show statistics
            cursor = conn.execute("SELECT COUNT(*) FROM loan_data")
            total_count = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT status, COUNT(*) FROM loan_data GROUP BY status")
            status_counts = cursor.fetchall()
            
            print(f"\n📈 Database Statistics:")
            print(f"   Total Loans: {total_count}")
            for status, count in status_counts:
                print(f"   {status}: {count}")
                
        except sqlite3.Error as e:
            print(f"❌ Error displaying records: {e}")
    
    def seed_database(self, num_records: int = 50, force_recreate: bool = False) -> None:
        """Main method to seed the database."""
        print("🌱 Starting Loan Data Database Seeding...")
        print(f"📍 Database path: {self.db_path}")
        
        # Remove existing database if force recreate
        if force_recreate and self.db_path.exists():
            os.remove(self.db_path)
            print("🗑️  Removed existing database")
        
        # Create connection
        conn = self.create_connection()
        
        try:
            # Create table
            self.create_loan_data_table(conn)
            
            # Check if data already exists
            cursor = conn.execute("SELECT COUNT(*) FROM loan_data")
            existing_count = cursor.fetchone()[0]
            
            if existing_count > 0 and not force_recreate:
                print(f"⚠️  Database already contains {existing_count} records")
                print("   Use --force to recreate the database")
                self.display_sample_records(conn)
                return
            
            # Generate and insert sample data
            print(f"📝 Generating {num_records} sample loan records...")
            sample_data = self.generate_sample_data(num_records)
            self.insert_sample_data(conn, sample_data)
            
            # Create indexes
            self.create_indexes(conn)
            
            # Display sample records
            self.display_sample_records(conn)
            
            print(f"\n🎉 Database seeding completed successfully!")
            print(f"📁 Database location: {self.db_path}")
            
        finally:
            conn.close()

def main():
    """Main function to run the seeder."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Loan Data Database Seeder")
    parser.add_argument("--records", "-r", type=int, default=50, 
                       help="Number of sample records to generate (default: 50)")
    parser.add_argument("--force", "-f", action="store_true",
                       help="Force recreate database (removes existing data)")
    
    args = parser.parse_args()
    
    seeder = LoanDataSeeder()
    seeder.seed_database(num_records=args.records, force_recreate=args.force)

if __name__ == "__main__":
    main()