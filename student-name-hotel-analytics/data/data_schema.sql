-- DROP TABLE IF EXISTS authors;
-- DROP TABLE IF EXISTS hotels;
-- DROP TABLE IF EXISTS reviews;

CREATE TABLE IF NOT EXISTS authors (
author_no INTEGER PRIMARY KEY AUTOINCREMENT,
author_id TEXT,
author_name TEXT,
author_location TEXT,
author_num_reviews INTEGER,
author_num_cities INTEGER,
author_num_helpful_votes INTEGER,
author_num_type_reviews INTEGER
);

CREATE TABLE IF NOT EXISTS hotels (
offering_id INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS reviews (
id INTEGER PRIMARY KEY,
author_no INTEGER,
author_id TEXT,
offering_id INTEGER,
overall REAL,
service REAL,
cleanliness REAL,
value REAL,
location_rating REAL,
sleep_quality REAL,
rooms REAL,
title TEXT,
text TEXT,
review_date DATE,
date_stayed TEXT,
via_mobile BOOLEAN,
author_num_helpful_votes INTEGER,

FOREIGN KEY(author_no) REFERENCES authors(author_no),
FOREIGN KEY(offering_id) REFERENCES hotels(offering_id)
);