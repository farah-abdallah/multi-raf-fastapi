# src/search/web.py

import requests
import json
import re
from urllib.parse import quote
from langchain_community.tools import DuckDuckGoSearchResults

def parse_search_results(results_string):
    try:
        results = json.loads(results_string)
        return [(result.get('title', 'Untitled'), result.get('link', '')) for result in results]
    except json.JSONDecodeError:
        print("Error parsing search results. Returning empty list.")
        return []

def simple_parse_html(html_content):
    """Simple HTML parsing without BeautifulSoup"""
    try:
        import re
        
        # Simple regex to extract basic information
        title_pattern = r'<a[^>]*class="result__a"[^>]*>([^<]+)</a>'
        snippet_pattern = r'<a[^>]*class="result__snippet"[^>]*>([^<]+)</a>'
        
        titles = re.findall(title_pattern, html_content)
        snippets = re.findall(snippet_pattern, html_content)
        
        results = []
        for i, (title, snippet) in enumerate(zip(titles[:3], snippets[:3])):
            results.append({
                'title': title.strip(),
                'snippet': snippet.strip(),
                'link': ''
            })
        
        return results
        
    except Exception as e:
        print(f"Simple parsing failed: {e}")
        return []
    

def parse_duckduckgo_html(html_content):
        """Parse DuckDuckGo HTML results"""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            results = []
            # Look for result containers
            result_divs = soup.find_all('div', class_='result')
            
            for div in result_divs[:5]:  # Limit to first 5 results
                try:
                    # Extract title
                    title_link = div.find('a', class_='result__a')
                    title = title_link.get_text(strip=True) if title_link else "Untitled"
                    link = title_link.get('href', '') if title_link else ''
                    
                    # Extract snippet
                    snippet_elem = div.find('a', class_='result__snippet')
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ''
                    
                    if title and snippet:
                        results.append({
                            'title': title,
                            'snippet': snippet,
                            'link': link
                        })
                except Exception as e:
                    print(f"Error parsing individual result: {e}")
                    continue
            
            return results
            
        except ImportError:
            print("BeautifulSoup not available, falling back to simple parsing")
            # Simple fallback parsing without BeautifulSoup
            return simple_parse_html(html_content)
        except Exception as e:
            print(f"Error parsing HTML: {e}")
            return []
    
def safe_web_search(query, max_results=3):
        """Improved web search with proper error handling"""
        try:
            # Clean and encode the query
            clean_query = query.strip().replace('\n', ' ')[:200]  # Limit length
            encoded_query = quote(clean_query)
            
            # Use proper headers to avoid blocking
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Referer': 'https://duckduckgo.com/',
            }
            
            # Try DuckDuckGo with improved parameters
            url = "https://html.duckduckgo.com/html"
            params = {
                'q': clean_query,
                'b': '',
                'kl': 'us-en',  # Change from 'wt-wt' to avoid issues
                'df': 'y'
            }
            
            print(f"Searching for: {clean_query}")
            response = requests.get(url, params=params, headers=headers, timeout=15)
            
            if response.status_code == 200 and response.content:
                # Parse results from HTML response
                results = parse_duckduckgo_html(response.text)
                if results:
                    print(f"Found {len(results)} web results")
                    return results[:max_results]
                else:
                    print("No results found in response")
                    return []
            else:
                print(f"Web search failed: Status {response.status_code}")
                return []
                
        except requests.RequestException as e:
            print(f"Network error during web search: {e}")
            return []
        except Exception as e:
            print(f"Web search error: {e}")
            return []
    
def perform_web_search(query, web_search_enabled=True, search_tool=None, rewrite_query_fn=None, knowledge_refinement_fn=None, parse_search_results_fn=None):
        """Improved web search with robust debug logging and user-facing error handling"""
        # Check if web search is disabled
        if not web_search_enabled:
            print("Web search disabled via configuration")
            return ["Web search disabled"], [("Configuration", "")]
        try:
            # Try improved web search first
            web_results = safe_web_search(query)
            print("\n[CRAG DEBUG] Raw web search results:", web_results)
            if web_results:
                # Process successful results
                web_knowledge = []
                sources = []
                for result in web_results:
                    if isinstance(result, dict):
                        title = result.get('title', 'Untitled')
                        content = result.get('snippet', result.get('content', ''))
                        link = result.get('link', result.get('url', ''))
                        if content:
                            web_knowledge.append(content)
                            # Always label as web search
                            sources.append((f"Web search: {title}", link))
                if web_knowledge:
                    print(f"[CRAG DEBUG] Parsed web knowledge: {web_knowledge}")
                    print(f"[CRAG DEBUG] Web sources: {sources}")
                    return web_knowledge, sources
                else:
                    print("[CRAG DEBUG] No usable content in web results.")
            # Fallback to original method if improved search fails
            if search_tool and rewrite_query_fn and knowledge_refinement_fn and parse_search_results_fn:  # Only if search tool is available
                print("Falling back to original web search...")
                rewritten_query = rewrite_query_fn(query)
                web_results = search_tool.run(rewritten_query)
                print(f"[CRAG DEBUG] Fallback web search raw results: {web_results}")
                web_knowledge = knowledge_refinement_fn(web_results)
                sources = [(f"Web search: {title}", link) for title, link in parse_search_results(web_results)]
                if not web_knowledge or all(not w.strip() for w in web_knowledge):
                    print("[CRAG DEBUG] Fallback web search returned no usable knowledge.")
                    return ["Web search failed to return any results relevant to your query."], [("Web Search Failure", "")]
                return web_knowledge, sources
            else:
                print("No search tool available")
                return ["Web search unavailable"], [("No Search Tool", "")]
        except Exception as e:
            print(f"Web search failed: {str(e)}")
            # Return empty results instead of crashing
            return [f"Web search unavailable at this time. Error: {str(e)}"], [("Web Search Error", "")]
    