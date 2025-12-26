import unittest
from unittest.mock import MagicMock, patch
import json
from wiki_crawler import WikiCrawler, WikipediaAPIError

class TestWikiCrawler(unittest.TestCase):
    def setUp(self):
        self.crawler = WikiCrawler(sleep_between_requests=0, verbose=False)

    @patch('requests.Session.get')
    def test_resolve_title_success(self, mock_get):
        # Mock response for title resolution
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "query": {
                "pages": [
                    {"title": "Python (programming language)", "pageid": 23862}
                ]
            }
        }
        mock_get.return_value = mock_response

        resolved = self.crawler.resolve_title("Python")
        self.assertEqual(resolved, "Python (programming language)")
        self.assertIn("Python", self.crawler.title_cache)

    @patch('requests.Session.get')
    def test_get_links(self, mock_get):
        # Mock response for links
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = [
            {
                "query": {
                    "pages": [{
                        "title": "A",
                        "links": [{"title": "B", "ns": 0}, {"title": "C", "ns": 0}]
                    }]
                }
            },
            {
                "query": {
                    "pages": [{"title": "A", "info": {}}] # For resolve_title
                }
            }
        ]
        mock_get.return_value = mock_response

        # We need to mock resolve_title or its API call
        with patch.object(WikiCrawler, 'resolve_title', return_value="A"):
            links = self.crawler.get_links("A")
            self.assertEqual(links, ["B", "C"])

    @patch.object(WikiCrawler, 'get_links')
    @patch.object(WikiCrawler, 'resolve_title')
    def test_find_path_bfs(self, mock_resolve, mock_get_links):
        mock_resolve.side_effect = lambda x: x
        mock_get_links.side_effect = lambda x: {
            "A": ["B", "C"],
            "B": ["D"],
            "C": ["D", "E"],
            "D": ["F"],
            "E": ["F"]
        }.get(x, [])

        path = self.crawler.find_path_bfs("A", "F")
        self.assertEqual(path, ["A", "B", "D", "F"]) # BFS finds shortest path

    @patch.object(WikiCrawler, 'get_links')
    @patch.object(WikiCrawler, 'get_linkshere')
    @patch.object(WikiCrawler, 'resolve_title')
    def test_find_path_bidi(self, mock_resolve, mock_get_linkshere, mock_get_links):
        mock_resolve.side_effect = lambda x: x
        mock_get_links.side_effect = lambda x: {
            "A": ["B"],
            "B": ["C"]
        }.get(x, [])
        mock_get_linkshere.side_effect = lambda x: {
            "C": ["B"],
            "B": ["A"]
        }.get(x, [])

        path = self.crawler.find_path_bidi("A", "C")
        self.assertEqual(path, ["A", "B", "C"])

    def test_title_score(self):
        score1 = self.crawler._title_score("Python", "Python (programming language)")
        score2 = self.crawler._title_score("Java", "Python (programming language)")
        self.assertGreater(score1, score2)

    def test_shorten_label(self):
        from wiki_crawler import _shorten_label
        self.assertEqual(_shorten_label("Short"), "Short")
        long_label = "This is a very long label that should be shortened"
        shortened = _shorten_label(long_label, max_len=20)
        self.assertTrue(len(shortened) <= 20)
        self.assertTrue("..." in shortened or " … " in shortened)

if __name__ == '__main__':
    unittest.main()