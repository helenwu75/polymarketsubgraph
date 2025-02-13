import os
import json
from typing import Dict, List, Optional
import importlib.util

class ProjectStructureAnalyzer:
    def __init__(self, root_dir: str):
        """Initialize the project structure analyzer.
        
        Args:
            root_dir (str): Root directory of the project
        """
        self.root_dir = root_dir
        self.structure = {}
        self.env_vars = {}
        self.dependencies = []
        self.python_files = []
        
    def analyze_structure(self) -> Dict:
        """Analyze the complete project structure."""
        # Get directory structure
        self.structure = self._get_directory_structure(self.root_dir)
        
        # Get environment variables (without sensitive values)
        self._analyze_env_file()
        
        # Get dependencies
        self._analyze_requirements()
        
        # Find all Python files
        self._find_python_files()
        
        # Generate complete project snapshot
        return {
            "directory_structure": self.structure,
            "env_variables": self.env_vars,
            "dependencies": self.dependencies,
            "python_files": self.python_files
        }
    
    def _get_directory_structure(self, startpath: str) -> Dict:
        """Recursively get the directory structure."""
        structure = {}
        
        # Skip common directories/files to ignore
        ignore = {'.git', '__pycache__', '.pytest_cache', '.venv', 'venv', '.env'}
        
        for root, dirs, files in os.walk(startpath):
            # Skip ignored directories
            dirs[:] = [d for d in dirs if d not in ignore]
            
            # Get relative path
            relative_path = os.path.relpath(root, startpath)
            if relative_path == '.':
                current = structure
            else:
                current = structure
                for part in relative_path.split(os.sep):
                    current = current.setdefault(part, {})
            
            # Add files
            for file in files:
                if not file.startswith('.') and file not in ignore:
                    current[file] = None
        
        return structure
    
    def _analyze_env_file(self) -> None:
        """Analyze .env file while protecting sensitive information."""
        env_path = os.path.join(self.root_dir, '.env')
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        try:
                            key = line.split('=')[0]
                            self.env_vars[key] = "[PROTECTED]"
                        except IndexError:
                            continue
    
    def _analyze_requirements(self) -> None:
        """Analyze requirements.txt file."""
        req_path = os.path.join(self.root_dir, 'requirements.txt')
        if os.path.exists(req_path):
            with open(req_path, 'r') as f:
                self.dependencies = [line.strip() for line in f if line.strip() 
                                  and not line.startswith('#')]
    
    def _find_python_files(self) -> None:
        """Find all Python files in the project."""
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, self.root_dir)
                    self.python_files.append(relative_path)

def save_project_snapshot(snapshot: Dict, output_file: str = 'project_snapshot.json') -> None:
    """Save the project snapshot to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(snapshot, f, indent=2)

def main():
    # Get the current directory
    current_dir = os.getcwd()
    
    # Create analyzer instance
    analyzer = ProjectStructureAnalyzer(current_dir)
    
    # Get project snapshot
    snapshot = analyzer.analyze_structure()
    
    # Save snapshot
    save_project_snapshot(snapshot)
    
    print("Project structure analysis complete. Results saved to project_snapshot.json")
    print("\nKey Statistics:")
    print(f"Total Python files: {len(snapshot['python_files'])}")
    print(f"Total dependencies: {len(snapshot['dependencies'])}")
    print(f"Environment variables: {len(snapshot['env_variables'])}")

if __name__ == "__main__":
    main()