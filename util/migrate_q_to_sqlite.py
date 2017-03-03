import sqlite3

conn = sqlite3.connect('q_blue.db')

c = conn.cursor()

# Create table
c.execute('''CREATE TABLE if not exists q_table
             (state_key text primary_key, q_values text not null, update_time, )''')

# Insert a row of data
c.execute("INSERT INTO stocks VALUES ('2006-01-05','BUY','RHAT',100,35.14)")

# Save (commit) the changes
conn.commit()

# We can also close the connection if we are done with it.
# Just be sure any changes have been committed or they will be lost.
conn.close()
