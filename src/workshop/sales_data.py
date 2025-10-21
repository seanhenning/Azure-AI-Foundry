import json
import logging
import os
from typing import Dict, List, Optional
from pathlib import Path
import re
import pandas as pd
from pdfminer.high_level import extract_text
from terminal_colors import TerminalColors as tc

DATASHEET_FILE = "datasheet/contoso-tents-datasheet.pdf"

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


class SalesData:
    _cached_data: Optional[Dict] = None
    
    def __init__(self: "SalesData") -> None:
        self._cached_data = None
        
    async def connect(self: "SalesData") -> None:
        """Load and parse the PDF datasheet."""
        if self._cached_data is not None:
            return
            
        try:
            # Use relative path from the current file
            current_dir = Path(__file__).parent
            pdf_path = current_dir / DATASHEET_FILE
            
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found at {pdf_path}")
            
            # Extract text from PDF
            text = extract_text(str(pdf_path))
            
            # Parse the text into structured data
            self._cached_data = self._parse_pdf_content(text)
            logger.debug("PDF data loaded and parsed.")
        except Exception as e:
            logger.exception("Error loading PDF data", exc_info=e)
            self._cached_data = None

    def _parse_pdf_content(self, text: str) -> Dict:
        """Parse PDF content into structured data."""
        data = {
            'products': [],
            'regions': set(),
            'product_types': set(),
            'categories': set(),
            'years': set()
        }
        
        # Split text into product sections (each starting with "Product Name:")
        product_sections = text.split('Product Name:')
        
        for section in product_sections[1:]:  # Skip first empty split
            product = {}
            lines = section.split('\n')
            
            # Initialize with product name
            product['product_name'] = lines[0].strip()
            
            # Process remaining lines
            current_key = None
            current_value = []
            
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue
                    
                if ':' in line:
                    # If we have a previous key, save it
                    if current_key:
                        product[current_key] = ' '.join(current_value).strip()
                        current_value = []
                    
                    # Start new key-value pair
                    key, value = line.split(':', 1)
                    current_key = key.strip().lower().replace(' ', '_')
                    current_value = [value.strip()]
                else:
                    # Continue previous value
                    current_value.append(line)
            
            # Save last key-value pair
            if current_key and current_value:
                product[current_key] = ' '.join(current_value).strip()
            
            # Extract and track unique values
            if 'region' in product:
                data['regions'].add(product['region'])
            if 'product_type' in product:
                data['product_types'].add(product['product_type'])
            if 'category' in product:
                data['categories'].add(product['category'])
            if product.get('year'):
                try:
                    data['years'].add(str(int(product['year'])))
                except ValueError:
                    pass
            
            data['products'].append(product)
        
        # Convert sets to sorted lists
        data['regions'] = sorted(list(data['regions']))
        data['product_types'] = sorted(list(data['product_types']))
        data['categories'] = sorted(list(data['categories']))
        data['years'] = sorted(list(data['years']))
        
        return data
    
    async def close(self: "SalesData") -> None:
        """Clear cached data."""
        self._cached_data = None
        logger.debug("PDF data cache cleared.")

    async def __get_regions(self: "SalesData") -> List[str]:
        """Return a list of unique regions from the PDF data."""
        return self._cached_data.get('regions', []) if self._cached_data else []

    async def __get_product_types(self: "SalesData") -> List[str]:
        """Return a list of unique product types from the PDF data."""
        return self._cached_data.get('product_types', []) if self._cached_data else []

    async def __get_product_categories(self: "SalesData") -> List[str]:
        """Return a list of unique product categories from the PDF data."""
        return self._cached_data.get('categories', []) if self._cached_data else []

    async def __get_reporting_years(self: "SalesData") -> List[str]:
        """Return a list of unique reporting years from the PDF data."""
        return self._cached_data.get('years', []) if self._cached_data else []

    async def get_data_info(self: "SalesData") -> str:
        """Return information about available data fields and values."""
        if not self._cached_data:
            return "No data available. Please ensure the PDF is loaded first."
            
        info = []
        # Show available fields from the first product
        if self._cached_data['products']:
            fields = sorted(self._cached_data['products'][0].keys())
            info.append(f"Available Fields: {', '.join(fields)}")
        
        regions = await self.__get_regions()
        product_types = await self.__get_product_types()
        product_categories = await self.__get_product_categories()
        reporting_years = await self.__get_reporting_years()

        info.extend([
            f"Regions: {', '.join(regions)}",
            f"Product Types: {', '.join(product_types)}",
            f"Product Categories: {', '.join(product_categories)}",
            f"Reporting Years: {', '.join(reporting_years)}",
            "\n"
        ])

        return "\n".join(info)

    async def async_fetch_sales_data(self: "SalesData", query_info: str) -> str:
        """
        This function answers questions about Contoso sales data by searching the PDF content.
        
        :param query_info: A description of what data to retrieve (e.g. "products in North region", "all camping tents")
        :return: Return data in JSON serializable format
        :rtype: str
        """
        print(f"\n{tc.BLUE}Function Call Tools: async_fetch_sales_data{tc.RESET}\n")
        print(f"{tc.BLUE}Processing query: {query_info}{tc.RESET}\n")
        
        if not self._cached_data or not self._cached_data.get('products'):
            return json.dumps("No data available. Please ensure the PDF is loaded first.")
            
        try:
            # Clean and tokenize the query
            query_terms = set(term.lower() for term in query_info.lower().split())
            
            # Filter products based on the query
            filtered_products = []
            for product in self._cached_data['products']:
                # Create searchable text from product fields
                product_text = ' '.join(str(v).lower() for v in product.values())
                
                # Calculate how many query terms match
                matching_terms = sum(1 for term in query_terms if term in product_text)
                
                # If any terms match, include the product with a match score
                if matching_terms > 0:
                    product_copy = product.copy()
                    product_copy['_match_score'] = matching_terms / len(query_terms)
                    filtered_products.append(product_copy)
            
            # Sort by match score and remove the score field
            filtered_products.sort(key=lambda p: p['_match_score'], reverse=True)
            for p in filtered_products:
                del p['_match_score']
            
            if not filtered_products:
                # Get available categories and types for suggestions
                categories = set()
                types = set()
                for p in self._cached_data['products']:
                    if 'category' in p:
                        categories.add(p['category'].lower())
                    if 'product_type' in p:
                        types.add(p['product_type'].lower())
                
                suggestions = []
                if categories:
                    suggestions.append(f"Categories: {', '.join(sorted(categories))}")
                if types:
                    suggestions.append(f"Product types: {', '.join(sorted(types))}")
                
                return json.dumps({
                    "found": 0,
                    "products": [],
                    "suggestion": "No products found. Try searching for: " + "; ".join(suggestions),
                    "status": "no_results"
                })
            
            # Format response as a structured dictionary
            result = {
                "found": len(filtered_products),
                "products": filtered_products,
                "fields": list(filtered_products[0].keys()),
                "message": f"Found {len(filtered_products)} matching products.",
                "status": "success"
            }
            return json.dumps(result)

        except Exception as e:
            return json.dumps({"Query failed with error": str(e), "query": query_info})
