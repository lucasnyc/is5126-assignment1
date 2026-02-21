CREATE INDEX IF NOT EXISTS idx_reviews_date ON reviews(review_date);
CREATE INDEX IF NOT EXISTS idx_reviews_user ON reviews(author_id);
CREATE INDEX IF NOT EXISTS idx_reviews_hotel ON reviews(offering_id);
