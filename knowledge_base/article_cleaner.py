"""
Article Cleaner - Clean and normalize article text
"""

import json
import re
import logging
from pathlib import Path
from typing import Dict, List
from bs4 import BeautifulSoup
from tqdm import tqdm
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArticleCleaner:
    """Clean and normalize article content"""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.articles_dir = Path(self.config['knowledge_base']['articles_dir'])

        # Common unwanted patterns
        self.unwanted_patterns = [
            r'\[edit\]',  # Wikipedia edit links
            r'\[\d+\]',  # Reference numbers
            r'\{.*?\}',  # Template markers
            r'==\s*See also\s*==.*',  # See also sections
            r'==\s*References\s*==.*',  # References sections
            r'==\s*External links\s*==.*',  # External links
            r'Retrieved from.*',  # Wikipedia footers
            r'Categories:.*',  # Wikipedia categories
            r'Jump to navigation.*',  # Navigation
            r'This article needs.*',  # Wikipedia maintenance tags
        ]

        # Compile patterns
        self.compiled_patterns = [re.compile(p, re.DOTALL | re.IGNORECASE)
                                 for p in self.unwanted_patterns]

    def clean_all_articles(self):
        """Clean all articles in the corpus"""
        article_files = list(self.articles_dir.glob("**/*.json"))
        article_files = [f for f in article_files if f.name != "master_index.json"]

        logger.info(f"Cleaning {len(article_files)} articles")

        cleaned_count = 0
        failed_count = 0

        for article_path in tqdm(article_files, desc="Cleaning articles"):
            try:
                with open(article_path, 'r', encoding='utf-8') as f:
                    article = json.load(f)

                # Clean the article
                cleaned_article = self.clean_article(article)

                # Validate
                if self.validate_article(cleaned_article):
                    # Save cleaned version
                    with open(article_path, 'w', encoding='utf-8') as f:
                        json.dump(cleaned_article, f, indent=2, ensure_ascii=False)
                    cleaned_count += 1
                else:
                    logger.warning(f"Article {article['id']} failed validation")
                    failed_count += 1

            except Exception as e:
                logger.error(f"Error cleaning {article_path}: {e}")
                failed_count += 1

        logger.info(f"Cleaning complete: {cleaned_count} cleaned, {failed_count} failed")
        return cleaned_count, failed_count

    def clean_article(self, article: Dict) -> Dict:
        """Clean a single article"""
        content = article['content']

        # Remove HTML if any
        content = self.remove_html(content)

        # Remove unwanted patterns
        content = self.remove_unwanted_patterns(content)

        # Normalize whitespace
        content = self.normalize_whitespace(content)

        # Clean up punctuation
        content = self.clean_punctuation(content)

        # Split into sections
        sections = self.split_into_sections(content)

        # Clean each section
        sections = [self.clean_section(s) for s in sections]

        # Join sections
        content = '\n\n'.join(sections)

        # Update article
        article['content'] = content
        article['cleaned'] = True
        article['length'] = len(content)
        article['num_sections'] = len(sections)

        return article

    def remove_html(self, text: str) -> str:
        """Remove any HTML tags"""
        soup = BeautifulSoup(text, 'html.parser')
        return soup.get_text()

    def remove_unwanted_patterns(self, text: str) -> str:
        """Remove unwanted patterns"""
        for pattern in self.compiled_patterns:
            text = pattern.sub('', text)
        return text

    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace"""
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)

        # Replace multiple newlines with double newline
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

        # Remove leading/trailing whitespace from lines
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)

        return text.strip()

    def clean_punctuation(self, text: str) -> str:
        """Clean up punctuation"""
        # Fix multiple punctuation
        text = re.sub(r'([.!?])\1+', r'\1', text)

        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'([.,!?;:])\s*([.,!?;:])', r'\1\2', text)

        # Ensure space after punctuation
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)

        return text

    def split_into_sections(self, text: str) -> List[str]:
        """Split text into logical sections"""
        # Split by double newlines
        sections = text.split('\n\n')

        # Filter out very short sections
        sections = [s for s in sections if len(s.strip()) > 50]

        return sections

    def clean_section(self, section: str) -> str:
        """Clean a single section"""
        # Remove section headers that are just special characters
        if re.match(r'^[=\-*#]+$', section.strip()):
            return ''

        # Remove very short lines (likely artifacts)
        lines = section.split('\n')
        lines = [line for line in lines if len(line.strip()) > 10 or
                self.is_header(line)]

        section = '\n'.join(lines)

        return section.strip()

    def is_header(self, line: str) -> bool:
        """Check if line is a header"""
        line = line.strip()

        # Check for common header patterns
        if re.match(r'^#{1,6}\s+', line):  # Markdown headers
            return True
        if line.isupper() and len(line) < 100:  # All caps short line
            return True
        if re.match(r'^[A-Z][a-z\s]+:$', line):  # Title Case with colon
            return True

        return False

    def validate_article(self, article: Dict) -> bool:
        """Validate cleaned article"""
        content = article['content']

        # Check minimum length
        if len(content) < 200:
            logger.debug(f"Article {article['id']} too short: {len(content)} chars")
            return False

        # Check for reasonable word count
        words = content.split()
        if len(words) < 50:
            logger.debug(f"Article {article['id']} too few words: {len(words)}")
            return False

        # Check for reasonable sentence count
        sentences = re.split(r'[.!?]+', content)
        sentences = [s for s in sentences if len(s.strip()) > 10]
        if len(sentences) < 3:
            logger.debug(f"Article {article['id']} too few sentences: {len(sentences)}")
            return False

        # Check readability - not too many special characters
        special_chars = len(re.findall(r'[^a-zA-Z0-9\s.,!?;:\-\(\)]', content))
        special_ratio = special_chars / len(content)
        if special_ratio > 0.1:
            logger.debug(f"Article {article['id']} too many special chars: {special_ratio:.2%}")
            return False

        return True

    def get_cleaning_stats(self) -> Dict:
        """Get statistics about cleaned articles"""
        article_files = list(self.articles_dir.glob("**/*.json"))
        article_files = [f for f in article_files if f.name != "master_index.json"]

        total_articles = len(article_files)
        cleaned_articles = 0
        total_length = 0
        total_sections = 0

        for article_path in article_files:
            try:
                with open(article_path, 'r', encoding='utf-8') as f:
                    article = json.load(f)

                if article.get('cleaned', False):
                    cleaned_articles += 1
                    total_length += article.get('length', 0)
                    total_sections += article.get('num_sections', 0)

            except Exception as e:
                logger.error(f"Error reading {article_path}: {e}")

        avg_length = total_length / cleaned_articles if cleaned_articles > 0 else 0
        avg_sections = total_sections / cleaned_articles if cleaned_articles > 0 else 0

        return {
            'total_articles': total_articles,
            'cleaned_articles': cleaned_articles,
            'avg_length': avg_length,
            'avg_sections': avg_sections,
            'total_length': total_length
        }


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Clean article corpus')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--stats', action='store_true',
                       help='Show cleaning statistics')

    args = parser.parse_args()

    cleaner = ArticleCleaner(args.config)

    if args.stats:
        stats = cleaner.get_cleaning_stats()
        print("\nCleaning Statistics:")
        print(f"  Total articles: {stats['total_articles']}")
        print(f"  Cleaned articles: {stats['cleaned_articles']}")
        print(f"  Average length: {stats['avg_length']:.0f} chars")
        print(f"  Average sections: {stats['avg_sections']:.1f}")
        print(f"  Total corpus size: {stats['total_length']:,} chars")
    else:
        cleaned, failed = cleaner.clean_all_articles()
        print(f"\n✓ Cleaned {cleaned} articles")
        if failed > 0:
            print(f"✗ {failed} articles failed validation")


if __name__ == "__main__":
    main()
