import sqlite3

DB_PATH = 'data/healthcare.db'

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

# Create tables
c.execute('''CREATE TABLE IF NOT EXISTS doctors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    specialization TEXT,
    location TEXT,
    contact TEXT
)''')

c.execute('''CREATE TABLE IF NOT EXISTS pharmacies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT,
    contact TEXT
)''')

c.execute('''CREATE TABLE IF NOT EXISTS patient_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    symptoms TEXT,
    predicted_disease TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)''')

# Add sample data
doctors = [
    ("Dr. Meera Rao", "Cardiology", "Mumbai", "9876543210"),
    ("Dr. Arjun Patel", "Dermatology", "Ahmedabad", "9876501234"),
    ("Dr. Kavita Sharma", "General Medicine", "Delhi", "9000012345"),
    ("Dr. Neel Joshi", "Endocrinology", "Pune", "9812345678"),
]

pharmacies = [
    ("HealthPlus Pharmacy", "Mumbai", "022-45454545"),
    ("WellCare Chemist", "Delhi", "011-43434343"),
    ("LifeLine Pharmacy", "Pune", "020-56565656"),
    ("Aarogya Medico", "Ahmedabad", "079-23232323"),
]

c.executemany("INSERT INTO doctors (name, specialization, location, contact) VALUES (?, ?, ?, ?)", doctors)
c.executemany("INSERT INTO pharmacies (name, location, contact) VALUES (?, ?, ?)", pharmacies)

conn.commit()
conn.close()

print("✅ Database seeded successfully!")