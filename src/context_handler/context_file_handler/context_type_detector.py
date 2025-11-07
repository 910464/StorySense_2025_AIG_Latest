from pathlib import Path
from typing import List, Dict, Any


class ContextTypeDetector:
    """Determines context type heuristically from file path and content metadata."""

    def determine(self, file_path: str, documents: List[Dict[str, Any]]) -> str:
        file_name = Path(file_path).name.lower()
        file_dir = Path(file_path).parent.name.lower()

        # Check directory structure first
        if 'business' in file_dir or 'rules' in file_dir:
            return 'business_rules'
        elif 'requirement' in file_dir or 'spec' in file_dir:
            return 'requirements'
        elif 'doc' in file_dir or 'guide' in file_dir or 'manual' in file_dir:
            return 'documentation'
        elif 'policy' in file_dir or 'procedure' in file_dir or 'guideline' in file_dir:
            return 'policies'
        elif 'example' in file_dir or 'template' in file_dir or 'sample' in file_dir:
            return 'examples'
        elif 'glossary' in file_dir or 'term' in file_dir or 'definition' in file_dir:
            return 'glossary'

        # Check file name
        if any(keyword in file_name for keyword in ['business', 'rule', 'logic', 'constraint']):
            return 'business_rules'
        elif any(keyword in file_name for keyword in ['requirement', 'spec', 'functional', 'non-functional']):
            return 'requirements'
        elif any(keyword in file_name for keyword in ['doc', 'guide', 'manual', 'readme']):
            return 'documentation'
        elif any(keyword in file_name for keyword in ['policy', 'procedure', 'guideline', 'standard']):
            return 'policies'
        elif any(keyword in file_name for keyword in ['example', 'template', 'sample', 'demo']):
            return 'examples'
        elif any(keyword in file_name for keyword in ['glossary', 'term', 'definition', 'acronym']):
            return 'glossary'

        # Default to documentation if can't determine
        return 'documentation'
