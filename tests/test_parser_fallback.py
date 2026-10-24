
from unittest.mock import patch, MagicMock
from zettlecast.parser import parse_url

def test_parse_url_fallback():
    """
    Test that parse_url falls back to BeautifulSoup when trafilatura fails.
    """
    url = "https://example.com/spa"
    html_content = """
    <html>
        <head><title>SPA Title</title></head>
        <body>
            <script>console.log('loading');</script>
            <div id="app">
                <h1>Welcome to the SPA</h1>
                <p>This is some content that trafilatura might miss if mocked to fail.</p>
            </div>
        </body>
    </html>
    """
    
    # Mock httpx.get to return the HTML
    with patch("httpx.get") as mock_get:
        mock_response = MagicMock()
        mock_response.text = html_content
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        # Mock trafilatura.extract to return None (simulate failure)
        with patch("trafilatura.extract", return_value=None):
            
            result = parse_url(url)
            
            assert result.status == "success"
            assert result.note is not None
            assert "Welcome to the SPA" in result.note.full_text
            assert "This is some content" in result.note.full_text
            # Verify script content is removed
            assert "console.log" not in result.note.full_text
            
            print("Fallback test passed!")

if __name__ == "__main__":
    test_parse_url_fallback()


def test_tidy_web_markdown_drops_empty_headings_and_frees_images():
    from zettlecast.parser import tidy_web_markdown

    text = ("# Title\n\nBy **Carlo M. Cipolla**\n\n## Table of contents\n\n## Part one\n\n"
            "![Figure 1](https://x.org/f1.gif) The text goes on.\n\n## Book\n\n### Chapter\n\nWords.\n\n## Empty at the end\n")
    assert tidy_web_markdown(text) == (
        "# Title\n\nBy **Carlo M. Cipolla**\n\n## Part one\n\n![Figure 1](https://x.org/f1.gif)\n\n"
        "The text goes on.\n\n## Book\n\n### Chapter\n\nWords.")


def test_byline_author():
    from zettlecast.parser import byline_author

    assert byline_author("# Title\n\nBy **Carlo M. Cipolla**\n\nText") == "Carlo M. Cipolla"
    assert byline_author("# Title\n\nby the river we sat and talked for hours on end, long into the night") is None


def test_images_are_not_embedded():
    from zettlecast.chunker import create_chunks

    text = "Some paragraph about the basic graph of stupidity and bandits.\n\n![Figure 1 - The basic graph](https://x.org/f1.gif)"
    indexed = " ".join(c.text for c in create_chunks(text, "n"))
    assert "Figure 1 - The basic graph" in indexed and "x.org" not in indexed
