"""
Corpus Builder - Scrape and collect articles for knowledge base
"""

import json
import time
import logging
import requests
from pathlib import Path
from typing import List, Dict, Optional
from urllib.parse import quote
from bs4 import BeautifulSoup
from tqdm import tqdm
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CorpusBuilder:
    """Build article corpus from multiple sources"""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.articles_dir = Path(self.config['knowledge_base']['articles_dir'])
        self.articles_dir.mkdir(parents=True, exist_ok=True)

        self.categories = self.config['knowledge_base']['categories']
        self.article_id = 0
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; EducationalBot/1.0)'
        })

    def build_corpus(self, target_articles: int = 100):
        """Build complete corpus from all sources"""
        logger.info(f"Building corpus with target of {target_articles} articles")

        articles_per_category = target_articles // len(self.categories)

        all_articles = []
        for category in self.categories:
            logger.info(f"Processing category: {category}")
            category_articles = self.collect_category_articles(
                category,
                articles_per_category
            )
            all_articles.extend(category_articles)

            # Save category articles
            category_dir = self.articles_dir / category
            category_dir.mkdir(exist_ok=True)

            for article in category_articles:
                self.save_article(article, category_dir)

            # Rate limiting
            time.sleep(2)

        # Save master index
        self.save_master_index(all_articles)

        logger.info(f"Corpus building complete! Collected {len(all_articles)} articles")
        return all_articles

    def collect_category_articles(self, category: str, target: int) -> List[Dict]:
        """Collect articles for a specific category"""
        articles = []

        # Generate search queries for this category
        queries = self.generate_queries_for_category(category)

        for query in queries[:min(len(queries), target)]:
            try:
                article = self.search_and_fetch_article(query, category)
                if article:
                    articles.append(article)
                    if len(articles) >= target:
                        break
                time.sleep(1)  # Rate limiting
            except Exception as e:
                logger.error(f"Error fetching article for query '{query}': {e}")
                continue

        return articles

    def generate_queries_for_category(self, category: str) -> List[str]:
        """Generate search queries for a category"""
        query_templates = {
            'fire_starting': [
                'how to start a fire without matches',
                'friction fire making techniques',
                'bow drill fire starting guide',
                'fire starting with flint and steel',
                'feather stick fire starting',
                'hand drill fire technique',
                'fire lay configurations',
                'tinder preparation for fire',
                'fire starting in wet conditions',
                'emergency fire starting methods'
            ],
            'water_purification': [
                'water purification methods survival',
                'how to purify water in wilderness',
                'boiling water for purification',
                'water filtration techniques',
                'solar water disinfection SODIS',
                'chemical water purification tablets',
                'finding safe water sources',
                'water distillation methods',
                'emergency water treatment',
                'signs of contaminated water'
            ],
            'shelter_building': [
                'wilderness shelter construction',
                'how to build a lean-to shelter',
                'debris hut construction guide',
                'tarp shelter configurations',
                'emergency shelter building',
                'shelter location selection',
                'insulation for wilderness shelters',
                'weatherproofing shelters',
                'snow shelter construction',
                'desert shelter building'
            ],
            'first_aid': [
                'wilderness first aid basics',
                'treating cuts and wounds outdoors',
                'splinting broken bones',
                'treating hypothermia symptoms',
                'heat exhaustion treatment',
                'snake bite first aid',
                'allergic reaction treatment',
                'CPR instructions guide',
                'treating burns first aid',
                'dealing with shock emergency',
                'wound infection prevention',
                'treating sprains and strains',
                'altitude sickness treatment',
                'dehydration treatment',
                'treating bee stings'
            ],
            'navigation': [
                'compass navigation basics',
                'map reading techniques',
                'navigation without compass',
                'using stars for navigation',
                'natural navigation methods',
                'dead reckoning navigation',
                'terrain association techniques',
                'GPS navigation basics',
                'reading topographic maps',
                'finding direction with sun'
            ],
            'food_foraging': [
                'edible wild plants identification',
                'safe berry identification',
                'edible mushroom identification',
                'fishing techniques survival',
                'setting animal traps',
                'edible insects survival',
                'plant identification safety',
                'poisonous plants to avoid',
                'wild edibles by season',
                'emergency food sources',
                'purslane edible weed',
                'dandelion edible uses',
                'cattail plant uses'
            ],
            'signaling': [
                'emergency signaling methods',
                'signal fire construction',
                'mirror signaling technique',
                'ground to air signals',
                'whistle signals survival',
                'smoke signal techniques',
                'flare usage guide',
                'SOS signal meaning',
                'emergency signal patterns',
                'visual distress signals'
            ],
            'weather': [
                'reading weather signs nature',
                'predicting weather without tools',
                'cloud types and weather',
                'natural weather indicators',
                'weather pattern basics',
                'lightning safety rules',
                'preparing for storms',
                'reading barometric pressure',
                'wind patterns and weather',
                'animal behavior weather prediction'
            ],
            'knots_tools': [
                'essential survival knots',
                'bowline knot instructions',
                'clove hitch knot guide',
                'square knot uses',
                'taut line hitch tutorial',
                'figure eight knot guide',
                'making cordage from plants',
                'knife safety and use',
                'axe safety techniques',
                'improvised tools survival'
            ],
            'emergency_procedures': [
                'emergency action plan steps',
                'survival priorities rule of threes',
                'stop and think survival',
                'emergency communication methods',
                'attracting rescuer attention',
                'survival psychology',
                'staying calm emergency',
                'emergency decision making',
                'risk assessment survival',
                'emergency shelter priorities',
                'hypothermia prevention',
                'heat stroke prevention',
                'emergency water procurement',
                'signaling for rescue'
            ]
        }

        return query_templates.get(category, [f"{category} survival guide"])

    def search_and_fetch_article(self, query: str, category: str) -> Optional[Dict]:
        """Search for and fetch an article on the topic"""
        try:
            # Try Wikipedia first
            article = self.fetch_from_wikipedia(query, category)
            if article:
                return article

            # Fallback to WikiHow
            article = self.fetch_from_wikihow(query, category)
            if article:
                return article

            return None

        except Exception as e:
            logger.error(f"Error searching for '{query}': {e}")
            return None

    def fetch_from_wikipedia(self, query: str, category: str) -> Optional[Dict]:
        """Fetch article from Wikipedia"""
        try:
            # Wikipedia API search
            search_url = "https://en.wikipedia.org/w/api.php"
            search_params = {
                'action': 'query',
                'list': 'search',
                'srsearch': query,
                'format': 'json',
                'srlimit': 1
            }

            response = self.session.get(search_url, params=search_params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if not data.get('query', {}).get('search'):
                return None

            page_title = data['query']['search'][0]['title']

            # Fetch page content
            content_params = {
                'action': 'query',
                'prop': 'extracts',
                'explaintext': True,
                'titles': page_title,
                'format': 'json'
            }

            response = self.session.get(search_url, params=content_params, timeout=10)
            response.raise_for_status()
            data = response.json()

            pages = data.get('query', {}).get('pages', {})
            if not pages:
                return None

            page = list(pages.values())[0]
            content = page.get('extract', '')

            if len(content) < 200:  # Too short
                return None

            # Extract keywords from query
            keywords = [word.lower() for word in query.split()
                       if len(word) > 3 and word.lower() not in ['how', 'what', 'with', 'from', 'this', 'that']]

            article = {
                'id': self.article_id,
                'title': page_title,
                'content': content,
                'category': category,
                'keywords': keywords,
                'source': 'wikipedia',
                'url': f"https://en.wikipedia.org/wiki/{quote(page_title)}",
                'length': len(content)
            }

            self.article_id += 1
            return article

        except Exception as e:
            logger.debug(f"Wikipedia fetch failed for '{query}': {e}")
            return None

    def fetch_from_wikihow(self, query: str, category: str) -> Optional[Dict]:
        """Fetch article from WikiHow"""
        try:
            search_url = f"https://www.wikihow.com/wikiHowTo?search={quote(query)}"

            response = self.session.get(search_url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Find first result
            result = soup.find('a', class_='result_link')
            if not result:
                return None

            article_url = result.get('href')
            if not article_url.startswith('http'):
                article_url = f"https://www.wikihow.com{article_url}"

            # Fetch article
            response = self.session.get(article_url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            title_elem = soup.find('h1', class_='firstHeading')
            title = title_elem.text.strip() if title_elem else query

            # Extract main content
            steps = soup.find_all('div', class_='step')
            content_parts = []

            for step in steps:
                step_text = step.get_text(separator=' ', strip=True)
                if step_text:
                    content_parts.append(step_text)

            content = '\n\n'.join(content_parts)

            if len(content) < 200:
                return None

            keywords = [word.lower() for word in query.split()
                       if len(word) > 3 and word.lower() not in ['how', 'what', 'with', 'from', 'this', 'that']]

            article = {
                'id': self.article_id,
                'title': title,
                'content': content,
                'category': category,
                'keywords': keywords,
                'source': 'wikihow',
                'url': article_url,
                'length': len(content)
            }

            self.article_id += 1
            return article

        except Exception as e:
            logger.debug(f"WikiHow fetch failed for '{query}': {e}")
            return None

    def save_article(self, article: Dict, category_dir: Path):
        """Save article to disk"""
        filename = f"article_{article['id']:05d}.json"
        filepath = category_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(article, f, indent=2, ensure_ascii=False)

    def save_master_index(self, articles: List[Dict]):
        """Save master index of all articles"""
        index_path = self.articles_dir / "master_index.json"

        index = {
            'total_articles': len(articles),
            'articles': [
                {
                    'id': a['id'],
                    'title': a['title'],
                    'category': a['category'],
                    'source': a['source'],
                    'length': a['length']
                }
                for a in articles
            ]
        }

        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2)

        logger.info(f"Saved master index to {index_path}")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Build article corpus')
    parser.add_argument('--target', type=int, default=100,
                       help='Target number of articles')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')

    args = parser.parse_args()

    builder = CorpusBuilder(args.config)
    articles = builder.build_corpus(args.target)

    print(f"\n✓ Successfully collected {len(articles)} articles")
    print(f"✓ Articles saved to {builder.articles_dir}")


if __name__ == "__main__":
    main()
