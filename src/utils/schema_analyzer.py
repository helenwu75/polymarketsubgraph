# src/utils/schema_analyzer.py

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
        print("Initializing SchemaAnalyzer...")
        self.api_key = api_key
        
        # Create necessary directories
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.logs_dir = os.path.join(project_root, 'logs')
        os.makedirs(self.logs_dir, exist_ok=True)
        
        self.logger = self._setup_logger()
        self._initialize_subgraphs()
        self._validate_config()

    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger('SchemaAnalyzer')
        logger.setLevel(logging.INFO)
        
        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(os.path.join(self.logs_dir, 'schema_analysis.log'))
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

    # ... [rest of the class implementation stays the same] ...

    def analyze_all_subgraphs(self, output_dir: str = None):
        """
        Analyze all configured subgraphs and generate documentation.
        
        Args:
            output_dir: Optional output directory override
        """
        if output_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            output_dir = os.path.join(project_root, "docs", "schemas")
            
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