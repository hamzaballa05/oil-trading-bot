"""Tests for NLP sentiment analysis."""

from oil_bot.nlp.sentiment import SentimentAnalyzer


def test_positive_text():
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze("Oil prices surge to record high, excellent gains")
    assert result["polarity"] > 0


def test_negative_text():
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze("terrible disaster, awful losses, very bad decline")
    assert result["polarity"] < 0


def test_neutral_text():
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze("Oil trading volume reported today")
    assert -1 <= result["polarity"] <= 1


def test_polarity_range():
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze("Crude oil markets remain stable")
    assert -1 <= result["polarity"] <= 1
    assert 0 <= result["subjectivity"] <= 1


def test_score_between_minus1_and_plus1():
    """Polarity must always be in [-1, +1]."""
    analyzer = SentimentAnalyzer()
    texts = [
        "oil prices rise strongly",
        "market crash devastates investors",
        "trading volumes unchanged",
        "opec meeting scheduled",
    ]
    for text in texts:
        result = analyzer.analyze(text)
        assert -1 <= result["polarity"] <= 1