#!/usr/bin/env python3
import os
import json
from typing import Dict, List, Optional
import requests
from dataclasses import dataclass
import logging
from datetime import datetime
from dotenv import load_dotenv

@dataclass
class SubgraphInfo:
    name: str
    url: str
    description: str

@dataclass
class EntityField:
    name: str
    type: str
    description: str
    is_required: bool

@dataclass
class Entity:
    name: str
    fields: List[EntityField]
    description: str

class SchemaAnalyzer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.logger = self._setup_logger()
        
        # Initialize subgraphs with URLs from environment variables
        self.subgraphs = {
            'pnl': SubgraphInfo(
                'PNL Subgraph',
                os.getenv('SUBGRAPH_URL_PNL', ''),
                'Profit and Loss Data'
            ),
            'activity': SubgraphInfo(
                'Activity Subgraph',
                os.getenv('SUBGRAPH_URL_ACTIVITY', ''),
                'Market Activity Tracking'
            ),
            'orderbook': SubgraphInfo(
                'Orderbook Subgraph',
                os.getenv('SUBGRAPH_URL_ORDERBOOK', ''),
                'Order Book and Pricing Information'
            ),
            'positions': SubgraphInfo(
                'Positions Subgraph',
                os.getenv('SUBGRAPH_URL_POSITIONS', ''),
                'User Positions and Trading History'
            ),
            'open_interest': SubgraphInfo(
                'Open Interest Subgraph',
                os.getenv('SUBGRAPH_URL_OPEN_INTEREST', ''),
                'Market Open Interest and Liquidity Data'
            )
        }
        
        # Validate configuration
        self._validate_config()

    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger('SchemaAnalyzer')
        logger.setLevel(logging.INFO)
        
        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler('schema_analysis.log')
        c_handler.setLevel(logging.INFO)
        f_handler.setLevel(logging.INFO)
        
        # Create formatters and add it to handlers
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        c_format = logging.Formatter(format_str)
        f_format = logging.Formatter(format_str)
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)
        
        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
        
        return logger

    def _validate_config(self):
        """Validate the configuration settings."""
        if not self.api_key:
            raise ValueError("API key is not set")
            
        for subgraph_id, info in self.subgraphs.items():
            if not info.url:
                raise ValueError(f"URL for {subgraph_id} subgraph is not set in environment variables")
            
            # Log partial URL for verification (only show beginning and end)
            url_preview = f"{info.url[:30]}...{info.url[-30:]}"
            self.logger.info(f"Loaded {subgraph_id} URL: {url_preview}")
                
        self.logger.info("Configuration validated successfully")

    def get_schema(self, subgraph_url: str) -> Optional[Dict]:
        """
        Fetch the GraphQL schema for a given subgraph.
        """
        try:
            # GraphQL introspection query
            query = """
            query IntrospectionQuery {
              __schema {
                types {
                  name
                  description
                  fields {
                    name
                    description
                    type {
                      name
                      kind
                      ofType {
                        name
                        kind
                      }
                    }
                  }
                }
              }
            }
            """
            
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}'
            }
            
            response = requests.post(
                subgraph_url,
                headers=headers,
                json={'query': query}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"Failed to fetch schema: {response.status_code}")
                self.logger.error(f"Response: {response.text}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error fetching schema: {str(e)}")
            return None

    def analyze_schema(self, schema_data: Dict) -> List[Entity]:
        """
        Analyze the schema data and extract entity information.
        """
        entities = []
        
        try:
            types = schema_data['data']['__schema']['types']
            
            # Filter for entity types (skip internal GraphQL types)
            entity_types = [t for t in types if not t['name'].startswith('__') and 
                          t['fields'] is not None]
            
            for entity_type in entity_types:
                fields = []
                for field in entity_type['fields']:
                    field_type = self._get_field_type(field['type'])
                    is_required = self._is_field_required(field['type'])
                    
                    fields.append(EntityField(
                        name=field['name'],
                        type=field_type,
                        description=field['description'] or '',
                        is_required=is_required
                    ))
                
                entities.append(Entity(
                    name=entity_type['name'],
                    fields=fields,
                    description=entity_type['description'] or ''
                ))
                
        except Exception as e:
            self.logger.error(f"Error analyzing schema: {str(e)}")
            
        return entities

    def _get_field_type(self, type_info: Dict) -> str:
        """
        Recursively resolve the field type from the GraphQL type information.
        """
        if type_info.get('ofType') is None:
            return type_info.get('name', 'Unknown')
        
        base_type = self._get_field_type(type_info['ofType'])
        if type_info['kind'] == 'LIST':
            return f'[{base_type}]'
        return base_type

    def _is_field_required(self, type_info: Dict) -> bool:
        """
        Determine if a field is required (non-null).
        """
        if type_info['kind'] == 'NON_NULL':
            return True
        if type_info.get('ofType'):
            return self._is_field_required(type_info['ofType'])
        return False

    def generate_documentation(self, entities: List[Entity], subgraph_name: str) -> str:
        """
        Generate markdown documentation for the analyzed schema.
        """
        doc = f"# {subgraph_name} Schema Documentation\n\n"
        doc += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        for entity in sorted(entities, key=lambda x: x.name):
            doc += f"## {entity.name}\n\n"
            if entity.description:
                doc += f"{entity.description}\n\n"
            
            doc += "### Fields\n\n"
            doc += "| Field Name | Type | Required | Description |\n"
            doc += "|------------|------|----------|-------------|\n"
            
            for field in sorted(entity.fields, key=lambda x: x.name):
                doc += f"| {field.name} | `{field.type}` | {'Yes' if field.is_required else 'No'} | {field.description} |\n"
            
            doc += "\n"
            
        return doc

    def analyze_all_subgraphs(self, output_dir: str = "docs/schemas"):
        """
        Analyze all configured subgraphs and generate documentation.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for subgraph_id, info in self.subgraphs.items():
            self.logger.info(f"Analyzing {info.name}...")
            
            # Fetch and analyze schema
            schema_data = self.get_schema(info.url)
            if not schema_data:
                self.logger.error(f"Failed to fetch schema for {info.name}")
                continue
                
            entities = self.analyze_schema(schema_data)
            
            # Generate documentation
            doc = self.generate_documentation(entities, info.name)
            
            # Save documentation
            output_path = os.path.join(output_dir, f"{subgraph_id}_schema.md")
            with open(output_path, 'w') as f:
                f.write(doc)
            
            self.logger.info(f"Documentation generated: {output_path}")
            
            # Log some statistics
            self.logger.info(f"Found {len(entities)} entities in {info.name}")
            total_fields = sum(len(entity.fields) for entity in entities)
            self.logger.info(f"Total fields: {total_fields}")

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Load API key from environment variable
    api_key = os.getenv('GRAPH_API_KEY')
    if not api_key:
        print("Error: GRAPH_API_KEY environment variable not set")
        print("Please ensure your .env file exists and contains GRAPH_API_KEY")
        return
    
    print(f"Loaded API key: {api_key[:6]}...{api_key[-4:]}")
    
    # Initialize analyzer and run analysis
    analyzer = SchemaAnalyzer(api_key)
    analyzer.analyze_all_subgraphs()

if __name__ == "__main__":
    main()