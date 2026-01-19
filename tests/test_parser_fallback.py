
import pytest
from unittest.mock import patch, MagicMock
from zettlecast.parser import parse_url
from zettlecast.models import ProcessingResult

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
        with patch("trafilatura.extract", return_value=None) as mock_extract:
            
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
