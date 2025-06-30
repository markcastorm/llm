#!/usr/bin/env python3
"""
Folder-Scanning Runbook Intelligence Engine
Scans folders, builds knowledge base, extracts start URLs, and prepares for Site Intelligence
"""

import json
import re
import os
import ollama
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import base64
from pathlib import Path
import docx
import pandas as pd
import PyPDF2
from PIL import Image
import pytesseract
import sqlite3
import hashlib
from datetime import datetime

class BusinessDomain(Enum):
    FINANCIAL_EXCHANGE = "financial_exchange"
    FINANCIAL_TRADING = "financial_trading"
    GOVERNMENT_REGULATORY = "government_regulatory"
    ECOMMERCE = "ecommerce"
    NEWS_MEDIA = "news_media"
    GENERAL = "general"

@dataclass
class RunbookKnowledge:
    """Knowledge base entry for a processed runbook"""
    runbook_id: str
    folder_path: str
    primary_source_url: str
    dataset_name: str
    business_domain: str
    extraction_patterns: Dict[str, Any]
    data_targets: List[Dict[str, Any]]
    authentication_info: Optional[Dict[str, Any]]
    output_requirements: Dict[str, Any]
    complexity_score: float
    confidence_score: float
    processed_files: List[str]
    created_at: str
    ready_for_site_analysis: bool

@dataclass
class SiteAnalysisPreparation:
    """Prepared package for Site Intelligence Engine"""
    runbook_id: str
    target_url: str
    expected_data_patterns: List[Dict[str, Any]]
    navigation_hints: List[str]
    authentication_requirements: Optional[Dict[str, Any]]
    business_context: Dict[str, Any]
    success_criteria: Dict[str, Any]
    special_instructions: List[str]

class RunbookFolderScanner:
    """
    Autonomous Folder Scanner that builds knowledge base and prepares for Site Intelligence
    """
    
    def __init__(self, 
                 knowledge_base_path: str = "runbook_knowledge.db",
                 model_name: str = "codellama:34b-instruct"):
        self.model_name = model_name
        self.knowledge_base_path = knowledge_base_path
        self.init_knowledge_base()
        self.content_processors = self._init_content_processors()
        self.supported_extensions = {
            '.pdf', '.docx', '.doc', '.txt', '.md',
            '.xlsx', '.xls', '.csv', '.json',
            '.png', '.jpg', '.jpeg', '.gif',
            '.zip', '.html', '.xml'
        }
        
    def init_knowledge_base(self):
        """Initialize SQLite knowledge base"""
        self.conn = sqlite3.connect(self.knowledge_base_path)
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS runbooks (
                runbook_id TEXT PRIMARY KEY,
                folder_path TEXT,
                primary_source_url TEXT,
                dataset_name TEXT,
                business_domain TEXT,
                extraction_patterns TEXT,
                data_targets TEXT,
                authentication_info TEXT,
                output_requirements TEXT,
                complexity_score REAL,
                confidence_score REAL,
                processed_files TEXT,
                created_at TEXT,
                ready_for_site_analysis BOOLEAN
            )
        ''')
        self.conn.commit()
        print(f"📊 Knowledge base initialized: {self.knowledge_base_path}")
    
    def scan_runbook_folder(self, folder_path: str, force_rescan: bool = False) -> RunbookKnowledge:
        """
        Main entry point: Scan a folder and build complete runbook knowledge
        
        Args:
            folder_path: Path to folder containing runbook files
            force_rescan: Force re-processing even if already in knowledge base
        """
        print(f"📁 Scanning runbook folder: {folder_path}")
        
        # Generate unique ID for this runbook
        runbook_id = self._generate_runbook_id(folder_path)
        
        # Check if already processed (unless force_rescan)
        if not force_rescan:
            existing = self._get_existing_runbook(runbook_id)
            if existing:
                print(f"✅ Runbook already in knowledge base: {runbook_id}")
                return existing
        
        # Step 1: Discover and classify all files in folder
        discovered_files = self._discover_files(folder_path)
        print(f"📋 Discovered {len(discovered_files)} files")
        
        # Step 2: Process all files and extract content
        processed_files = self._process_all_files(discovered_files)
        print(f"✅ Processed {len(processed_files)} files successfully")
        
        # Step 3: Classify file roles and relationships
        file_classification = self._classify_file_roles(processed_files)
        
        # Step 4: Build unified content for LLM analysis
        unified_content = self._merge_file_contents(processed_files, file_classification)
        
        # Step 5: Extract requirements using LLM
        print("🧠 Analyzing runbook with LLM...")
        requirements = self._extract_requirements_with_llm(unified_content, folder_path)
        
        # Step 6: Build knowledge base entry
        knowledge = self._build_runbook_knowledge(
            runbook_id, folder_path, requirements, processed_files
        )
        
        # Step 7: Store in knowledge base
        self._store_in_knowledge_base(knowledge)
        
        # Step 8: Validate readiness for site analysis
        knowledge.ready_for_site_analysis = self._validate_site_analysis_readiness(knowledge)
        
        print(f"🎉 Runbook knowledge built: {knowledge.dataset_name}")
        print(f"🌐 Primary URL: {knowledge.primary_source_url}")
        print(f"✅ Ready for Site Analysis: {knowledge.ready_for_site_analysis}")
        
        return knowledge
    
    def prepare_for_site_analysis(self, runbook_id: str) -> Optional[SiteAnalysisPreparation]:
        """
        Prepare a specific runbook for Site Intelligence Engine
        """
        knowledge = self._get_existing_runbook(runbook_id)
        if not knowledge:
            print(f"❌ Runbook not found: {runbook_id}")
            return None
        
        if not knowledge.ready_for_site_analysis:
            print(f"⚠️ Runbook not ready for site analysis: {runbook_id}")
            return None
        
        # Extract key information for Site Intelligence
        preparation = SiteAnalysisPreparation(
            runbook_id=runbook_id,
            target_url=knowledge.primary_source_url,
            expected_data_patterns=knowledge.data_targets,
            navigation_hints=self._extract_navigation_hints(knowledge),
            authentication_requirements=knowledge.authentication_info,
            business_context={
                "domain": knowledge.business_domain,
                "dataset_name": knowledge.dataset_name,
                "complexity": knowledge.complexity_score
            },
            success_criteria=self._extract_success_criteria(knowledge),
            special_instructions=self._extract_special_instructions(knowledge)
        )
        
        print(f"🎯 Site Analysis preparation ready for: {knowledge.dataset_name}")
        return preparation
    
    def get_all_runbooks(self) -> List[RunbookKnowledge]:
        """Get all runbooks from knowledge base"""
        cursor = self.conn.execute("SELECT * FROM runbooks")
        runbooks = []
        for row in cursor.fetchall():
            knowledge = self._row_to_runbook_knowledge(row)
            runbooks.append(knowledge)
        return runbooks
    
    def search_runbooks(self, query: str) -> List[RunbookKnowledge]:
        """Search runbooks by dataset name, URL, or domain"""
        cursor = self.conn.execute("""
            SELECT * FROM runbooks 
            WHERE dataset_name LIKE ? OR primary_source_url LIKE ? OR business_domain LIKE ?
        """, (f"%{query}%", f"%{query}%", f"%{query}%"))
        
        runbooks = []
        for row in cursor.fetchall():
            knowledge = self._row_to_runbook_knowledge(row)
            runbooks.append(knowledge)
        return runbooks
    
    def _discover_files(self, folder_path: str) -> List[Dict[str, Any]]:
        """Recursively discover all relevant files in folder"""
        discovered_files = []
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            print(f"❌ Folder not found: {folder_path}")
            return discovered_files
        
        # Recursively find all supported files
        for file_path in folder_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                file_info = {
                    'path': str(file_path),
                    'name': file_path.name,
                    'extension': file_path.suffix.lower(),
                    'size': file_path.stat().st_size,
                    'relative_path': str(file_path.relative_to(folder_path)),
                    'is_likely_runbook': self._is_likely_runbook_file(file_path)
                }
                discovered_files.append(file_info)
        
        # Sort by likelihood of being main runbook
        discovered_files.sort(key=lambda x: (
            x['is_likely_runbook'],
            'runbook' in x['name'].lower(),
            x['size']
        ), reverse=True)
        
        return discovered_files
    
    def _is_likely_runbook_file(self, file_path: Path) -> bool:
        """Determine if file is likely the main runbook"""
        name_lower = file_path.name.lower()
        runbook_indicators = [
            'runbook', 'instruction', 'manual', 'guide', 
            'readme', 'requirements', 'spec'
        ]
        return any(indicator in name_lower for indicator in runbook_indicators)
    
    def _generate_runbook_id(self, folder_path: str) -> str:
        """Generate unique ID for runbook based on folder path and contents"""
        folder_hash = hashlib.md5(str(folder_path).encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d")
        return f"runbook_{timestamp}_{folder_hash}"
    
    def _get_existing_runbook(self, runbook_id: str) -> Optional[RunbookKnowledge]:
        """Check if runbook already exists in knowledge base"""
        cursor = self.conn.execute("SELECT * FROM runbooks WHERE runbook_id = ?", (runbook_id,))
        row = cursor.fetchone()
        if row:
            return self._row_to_runbook_knowledge(row)
        return None
    
    def _process_all_files(self, discovered_files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process all discovered files and extract content"""
        processed_files = []
        
        for file_info in discovered_files:
            try:
                file_ext = file_info['extension']
                
                if file_ext in self.content_processors:
                    content = self.content_processors[file_ext](file_info)
                    
                    processed_file = {
                        'original_info': file_info,
                        'extracted_content': content,
                        'file_type': file_ext,
                        'file_name': file_info['name'],
                        'relative_path': file_info['relative_path'],
                        'processing_status': 'success',
                        'content_length': len(content) if content else 0
                    }
                else:
                    # Try basic text processing
                    try:
                        with open(file_info['path'], 'r', encoding='utf-8') as f:
                            content = f.read()
                    except:
                        content = f"File type not supported: {file_ext}"
                    
                    processed_file = {
                        'original_info': file_info,
                        'extracted_content': content,
                        'file_type': file_ext,
                        'file_name': file_info['name'],
                        'relative_path': file_info['relative_path'],
                        'processing_status': 'fallback',
                        'content_length': len(content)
                    }
                
                processed_files.append(processed_file)
                print(f"✅ Processed: {processed_file['file_name']} ({processed_file['content_length']} chars)")
                
            except Exception as e:
                print(f"⚠️ Error processing {file_info['name']}: {e}")
                continue
        
        return processed_files
    
    def _classify_file_roles(self, processed_files: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """Classify files into roles within the runbook"""
        classification = {
            'primary_runbook': [],
            'supplementary_instructions': [],
            'screenshots': [],
            'sample_data': [],
            'metadata_specs': [],
            'configuration': [],
            'credentials': []
        }
        
        for file_info in processed_files:
            content = file_info['extracted_content'].lower()
            file_name = file_info['file_name'].lower()
            file_type = file_info['file_type']
            
            # Primary runbook detection
            if (file_info['original_info']['is_likely_runbook'] and 
                len(content) > 500 and
                any(keyword in content for keyword in ['dataset', 'source', 'instructions', 'runbook'])):
                classification['primary_runbook'].append(file_info)
            
            # Screenshots and images
            elif file_type in ['.png', '.jpg', '.jpeg', '.gif']:
                classification['screenshots'].append(file_info)
            
            # Configuration and credentials
            elif any(keyword in file_name for keyword in ['config', 'settings', 'credential', 'auth']):
                classification['credentials'].append(file_info)
            
            # Sample data
            elif any(keyword in file_name for keyword in ['sample', 'example', 'test']):
                classification['sample_data'].append(file_info)
            
            # Metadata specifications
            elif any(keyword in file_name for keyword in ['meta', 'spec', 'schema']):
                classification['metadata_specs'].append(file_info)
            
            # Everything else as supplementary
            else:
                classification['supplementary_instructions'].append(file_info)
        
        # Log classification results
        for category, files in classification.items():
            if files:
                file_names = [f['file_name'] for f in files]
                print(f"📋 {category}: {file_names}")
        
        return classification
    
    def _merge_file_contents(self, processed_files: List[Dict], classification: Dict) -> Dict[str, str]:
        """Merge file contents for comprehensive LLM analysis"""
        unified_content = {
            'primary_runbook_content': '',
            'supplementary_instructions': '',
            'screenshot_descriptions': '',
            'sample_data_content': '',
            'configuration_info': '',
            'all_content_combined': ''
        }
        
        # Primary runbook (highest priority)
        for file_info in classification['primary_runbook']:
            unified_content['primary_runbook_content'] += f"\n\n=== MAIN RUNBOOK: {file_info['file_name']} ===\n"
            unified_content['primary_runbook_content'] += file_info['extracted_content']
        
        # Supplementary instructions
        for file_info in classification['supplementary_instructions']:
            unified_content['supplementary_instructions'] += f"\n\n=== SUPPLEMENTARY: {file_info['file_name']} ===\n"
            unified_content['supplementary_instructions'] += file_info['extracted_content']
        
        # Screenshots and visual references
        for file_info in classification['screenshots']:
            unified_content['screenshot_descriptions'] += f"\n\n=== SCREENSHOT: {file_info['file_name']} ===\n"
            unified_content['screenshot_descriptions'] += file_info['extracted_content']
        
        # Configuration and credentials
        for file_info in classification['credentials']:
            unified_content['configuration_info'] += f"\n\n=== CONFIG: {file_info['file_name']} ===\n"
            unified_content['configuration_info'] += file_info['extracted_content']
        
        # Sample data
        for file_info in classification['sample_data']:
            unified_content['sample_data_content'] += f"\n\n=== SAMPLE: {file_info['file_name']} ===\n"
            unified_content['sample_data_content'] += file_info['extracted_content']
        
        # Combine everything for complete analysis
        all_parts = [
            unified_content['primary_runbook_content'],
            unified_content['supplementary_instructions'],
            unified_content['screenshot_descriptions'],
            unified_content['configuration_info'],
            unified_content['sample_data_content']
        ]
        unified_content['all_content_combined'] = '\n\n'.join(filter(None, all_parts))
        
        return unified_content
    
    def _extract_requirements_with_llm(self, unified_content: Dict[str, str], folder_path: str) -> Dict:
        """Extract comprehensive requirements using LLM"""
        
        prompt = f"""
        You are an expert runbook analyst building a knowledge base for autonomous web scraping.
        
        FOLDER PATH: {folder_path}
        
        MAIN RUNBOOK CONTENT:
        {unified_content['primary_runbook_content']}
        
        SUPPLEMENTARY INSTRUCTIONS:
        {unified_content['supplementary_instructions']}
        
        CONFIGURATION INFO:
        {unified_content['configuration_info']}
        
        SCREENSHOT DESCRIPTIONS:
        {unified_content['screenshot_descriptions']}
        
        SAMPLE DATA:
        {unified_content['sample_data_content']}
        
        Extract and structure ALL information needed for autonomous web scraping:
        
        1. DATASET IDENTIFICATION:
           - Dataset name/abbreviation
           - Business description
           - Data source organization
        
        2. PRIMARY SOURCE URL:
           - Main website to scrape (look for "Link to source", "URL", "website")
           - Login/authentication URLs if different
        
        3. AUTHENTICATION REQUIREMENTS:
           - Login credentials needed
           - API keys or tokens
           - Special authentication steps
        
        4. DATA TARGETS (be extremely specific):
           - Exact field names to extract
           - Data types (percentage, currency, date, text)
           - Business meaning of each field
           - Validation rules for each field
        
        5. EXTRACTION PATTERNS (critical for automation):
           - Text patterns that indicate data location
           - How to find effective dates
           - How to extract commodity/item names
           - How to handle multiple items in one sentence
           - Special cases like "restored to original", "remains at"
        
        6. NAVIGATION REQUIREMENTS:
           - Steps to reach the data
           - Search functionality usage
           - Pagination handling
           - Form interactions needed
        
        7. OUTPUT SPECIFICATIONS:
           - Required file formats (XLS, CSV, ZIP)
           - File naming conventions
           - Directory structure
           - Metadata requirements
        
        8. BUSINESS LOGIC:
           - Data transformation rules
           - Calculation requirements
           - Quality validation rules
        
        9. FREQUENCY & TIMING:
           - How often to run
           - Specific timing requirements
           - SLA deadlines
        
        10. SPECIAL INSTRUCTIONS:
            - Unique handling requirements
            - Error handling preferences
            - Quality standards
        
        Respond in structured JSON format with these exact keys:
        - dataset_name
        - dataset_description
        - primary_source_url
        - authentication_requirements
        - data_targets
        - extraction_patterns
        - navigation_requirements
        - output_specifications
        - business_logic_rules
        - temporal_requirements
        - special_instructions
        - business_domain
        - complexity_assessment
        - confidence_indicators
        
        Be extremely thorough and specific. This will be used for fully autonomous operation.
        """
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = response['message']['content']
            print(f"🤖 LLM Response: {len(response_text)} characters")
            
            # Extract JSON
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            
            if json_match:
                requirements = json.loads(json_match.group(0))
                print("✅ LLM extraction successful")
                return requirements
            else:
                print("⚠️ No JSON in LLM response, using fallback")
                return self._fallback_extraction(unified_content)
                
        except Exception as e:
            print(f"❌ LLM extraction failed: {e}")
            return self._fallback_extraction(unified_content)
    
    def _build_runbook_knowledge(self, runbook_id: str, folder_path: str, 
                                requirements: Dict, processed_files: List[Dict]) -> RunbookKnowledge:
        """Build comprehensive knowledge base entry"""
        
        return RunbookKnowledge(
            runbook_id=runbook_id,
            folder_path=folder_path,
            primary_source_url=requirements.get('primary_source_url', ''),
            dataset_name=requirements.get('dataset_name', Path(folder_path).name),
            business_domain=requirements.get('business_domain', 'general'),
            extraction_patterns=requirements.get('extraction_patterns', {}),
            data_targets=requirements.get('data_targets', []),
            authentication_info=requirements.get('authentication_requirements'),
            output_requirements=requirements.get('output_specifications', {}),
            complexity_score=self._calculate_complexity(requirements),
            confidence_score=self._calculate_confidence(requirements, processed_files),
            processed_files=[f['file_name'] for f in processed_files],
            created_at=datetime.now().isoformat(),
            ready_for_site_analysis=False  # Will be set after validation
        )
    
    def _store_in_knowledge_base(self, knowledge: RunbookKnowledge):
        """Store runbook knowledge in database"""
        self.conn.execute("""
            INSERT OR REPLACE INTO runbooks VALUES 
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            knowledge.runbook_id,
            knowledge.folder_path,
            knowledge.primary_source_url,
            knowledge.dataset_name,
            knowledge.business_domain,
            json.dumps(knowledge.extraction_patterns),
            json.dumps(knowledge.data_targets),
            json.dumps(knowledge.authentication_info) if knowledge.authentication_info else None,
            json.dumps(knowledge.output_requirements),
            knowledge.complexity_score,
            knowledge.confidence_score,
            json.dumps(knowledge.processed_files),
            knowledge.created_at,
            knowledge.ready_for_site_analysis
        ))
        self.conn.commit()
        print(f"💾 Stored in knowledge base: {knowledge.runbook_id}")
    
    def _validate_site_analysis_readiness(self, knowledge: RunbookKnowledge) -> bool:
        """Validate if runbook is ready for Site Intelligence Engine"""
        validation_checks = [
            ('Primary URL', bool(knowledge.primary_source_url)),
            ('Data Targets', bool(knowledge.data_targets)),
            ('Extraction Patterns', bool(knowledge.extraction_patterns)),
            ('Dataset Name', bool(knowledge.dataset_name)),
            ('Business Domain', bool(knowledge.business_domain))
        ]
        
        passed_checks = 0
        for check_name, passed in validation_checks:
            if passed:
                passed_checks += 1
                print(f"✅ {check_name}: PASS")
            else:
                print(f"❌ {check_name}: FAIL")
        
        readiness_score = passed_checks / len(validation_checks)
        is_ready = readiness_score >= 0.8  # Need 80% validation success
        
        print(f"🎯 Site Analysis Readiness: {readiness_score:.1%} ({'READY' if is_ready else 'NOT READY'})")
        return is_ready
    
    def _extract_navigation_hints(self, knowledge: RunbookKnowledge) -> List[str]:
        """Extract navigation hints for Site Intelligence"""
        hints = []
        
        # Extract from extraction patterns
        if knowledge.extraction_patterns:
            patterns = knowledge.extraction_patterns
            if 'navigation_requirements' in patterns:
                hints.extend(patterns['navigation_requirements'])
        
        # Add domain-specific hints
        if 'exchange' in knowledge.business_domain:
            hints.extend([
                "Look for notice/announcement sections",
                "Check for margin ratio or trading updates",
                "Navigate to public notices or regulatory updates"
            ])
        
        return hints
    
    def _extract_success_criteria(self, knowledge: RunbookKnowledge) -> Dict[str, Any]:
        """Extract success criteria for validation"""
        criteria = {
            "expected_data_fields": len(knowledge.data_targets),
            "minimum_confidence": 0.8,
            "required_extractions": []
        }
        
        for target in knowledge.data_targets:
            if isinstance(target, dict):
                criteria["required_extractions"].append(target.get("field_name", "unknown"))
        
        return criteria
    
    def _extract_special_instructions(self, knowledge: RunbookKnowledge) -> List[str]:
        """Extract special instructions for Site Intelligence"""
        instructions = []
        
        # Add authentication instructions
        if knowledge.authentication_info:
            instructions.append("Authentication required - check credentials")
        
        # Add domain-specific instructions
        if 'financial' in knowledge.business_domain:
            instructions.extend([
                "Handle percentage data carefully",
                "Look for effective dates and margin adjustments",
                "Process multiple commodities in single sentences"
            ])
        
        return instructions
    
    def _calculate_complexity(self, requirements: Dict) -> float:
        """Calculate complexity score based on requirements"""
        factors = []
        
        # Authentication complexity
        if requirements.get('authentication_requirements'):
            factors.append(0.2)
        
        # Data targets complexity
        data_targets = requirements.get('data_targets', [])
        factors.append(min(len(data_targets) / 10.0, 0.3))
        
        # Navigation complexity
        nav_req = requirements.get('navigation_requirements', {})
        if nav_req:
            factors.append(0.2)
        
        # Special instructions complexity
        special = requirements.get('special_instructions', [])
        factors.append(min(len(special) / 5.0, 0.2))
        
        # Base complexity
        factors.append(0.1)
        
        return min(sum(factors), 1.0)
    
    def _calculate_confidence(self, requirements: Dict, processed_files: List[Dict]) -> float:
        """Calculate confidence in extracted requirements"""
        confidence_factors = []
        
        # URL extraction confidence
        if requirements.get('primary_source_url'):
            confidence_factors.append(0.25)
        
        # Data targets confidence
        if requirements.get('data_targets'):
            confidence_factors.append(0.25)
        
        # Extraction patterns confidence
        if requirements.get('extraction_patterns'):
            confidence_factors.append(0.25)
        
        # File processing confidence
        successful_files = sum(1 for f in processed_files if f.get('processing_status') == 'success')
        if successful_files > 0:
            confidence_factors.append(0.25)
        
        return min(sum(confidence_factors), 1.0)
    
    def _fallback_extraction(self, unified_content: Dict) -> Dict:
        """Fallback extraction using pattern matching"""
        all_content = unified_content['all_content_combined']
        
        # Extract URLs
        urls = re.findall(r'https?://[^\s]+', all_content)
        primary_url = urls[0] if urls else ''
        
        # Extract dataset name
        dataset_match = re.search(r'dataset[:\s]+([^\n]+)', all_content, re.IGNORECASE)
        dataset_name = dataset_match.group(1).strip() if dataset_match else 'Unknown'
        
        return {
            "dataset_name": dataset_name,
            "primary_source_url": primary_url,
            "data_targets": [],
            "extraction_patterns": {},
            "business_domain": "general",
            "complexity_assessment": "medium",
            "confidence_indicators": ["fallback_extraction_used"]
        }
    
    def _row_to_runbook_knowledge(self, row) -> RunbookKnowledge:
        """Convert database row to RunbookKnowledge object"""
        return RunbookKnowledge(
            runbook_id=row[0],
            folder_path=row[1],
            primary_source_url=row[2],
            dataset_name=row[3],
            business_domain=row[4],
            extraction_patterns=json.loads(row[5]) if row[5] else {},
            data_targets=json.loads(row[6]) if row[6] else [],
            authentication_info=json.loads(row[7]) if row[7] else None,
            output_requirements=json.loads(row[8]) if row[8] else {},
            complexity_score=row[9],
            confidence_score=row[10],
            processed_files=json.loads(row[11]) if row[11] else [],
            created_at=row[12],
            ready_for_site_analysis=bool(row[13])
        )
    
    def _init_content_processors(self) -> Dict[str, callable]:
        """Initialize file content processors"""
        return {
            ".pdf": self._process_pdf,
            ".docx": self._process_docx,
            ".doc": self._process_doc,
            ".xlsx": self._process_excel,
            ".xls": self._process_excel,
            ".csv": self._process_csv,
            ".txt": self._process_text,
            ".md": self._process_text,
            ".json": self._process_json,
            ".png": self._process_image,
            ".jpg": self._process_image,
            ".jpeg": self._process_image
        }
    
    # File processing methods (same as before but with file path handling)
    def _process_pdf(self, file_info: Dict) -> str:
        """Extract text from PDF"""
        try:
            with open(file_info['path'], 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ''
                for page in reader.pages:
                    text += page.extract_text()
                return text
        except Exception as e:
            print(f"⚠️ PDF processing failed for {file_info['name']}: {e}")
            return f"PDF processing failed: {file_info['name']}"
    
    def _process_docx(self, file_info: Dict) -> str:
        """Extract text from DOCX"""
        try:
            doc = docx.Document(file_info['path'])
            text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            print(f"⚠️ DOCX processing failed for {file_info['name']}: {e}")
            return f"DOCX processing failed: {file_info['name']}"
    
    def _process_doc(self, file_info: Dict) -> str:
        """Extract text from DOC (legacy format)"""
        try:
            # For .doc files, try basic text extraction
            with open(file_info['path'], 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return content
        except Exception as e:
            print(f"⚠️ DOC processing failed for {file_info['name']}: {e}")
            return f"DOC processing failed: {file_info['name']}"
    
    def _process_excel(self, file_info: Dict) -> str:
        """Extract text from Excel files"""
        try:
            # Read all sheets and combine
            xl_file = pd.ExcelFile(file_info['path'])
            all_content = []
            
            for sheet_name in xl_file.sheet_names:
                df = pd.read_excel(file_info['path'], sheet_name=sheet_name)
                all_content.append(f"=== SHEET: {sheet_name} ===")
                all_content.append(df.to_string())
            
            return '\n\n'.join(all_content)
        except Exception as e:
            print(f"⚠️ Excel processing failed for {file_info['name']}: {e}")
            return f"Excel processing failed: {file_info['name']}"
    
    def _process_csv(self, file_info: Dict) -> str:
        """Extract text from CSV"""
        try:
            df = pd.read_csv(file_info['path'])
            return df.to_string()
        except Exception as e:
            print(f"⚠️ CSV processing failed for {file_info['name']}: {e}")
            return f"CSV processing failed: {file_info['name']}"
    
    def _process_text(self, file_info: Dict) -> str:
        """Process text files (TXT, MD)"""
        try:
            with open(file_info['path'], 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            try:
                # Try with different encoding
                with open(file_info['path'], 'r', encoding='latin-1') as f:
                    return f.read()
            except Exception as e2:
                print(f"⚠️ Text processing failed for {file_info['name']}: {e2}")
                return f"Text processing failed: {file_info['name']}"
    
    def _process_json(self, file_info: Dict) -> str:
        """Process JSON files"""
        try:
            with open(file_info['path'], 'r', encoding='utf-8') as f:
                data = json.load(f)
            return json.dumps(data, indent=2)
        except Exception as e:
            print(f"⚠️ JSON processing failed for {file_info['name']}: {e}")
            return f"JSON processing failed: {file_info['name']}"
    
    def _process_image(self, file_info: Dict) -> str:
        """Extract text from images using OCR"""
        try:
            image = Image.open(file_info['path'])
            # Try OCR extraction
            text = pytesseract.image_to_string(image)
            if text.strip():
                return f"OCR Extracted from {file_info['name']}:\n{text}"
            else:
                return f"Screenshot reference: {file_info['name']} (no text detected)"
        except Exception as e:
            print(f"⚠️ Image OCR failed for {file_info['name']}: {e}")
            return f"Screenshot reference: {file_info['name']} (OCR failed)"

# =====================================================================
# DEMONSTRATION AND USAGE EXAMPLES
# =====================================================================

def demo_folder_scanning():
    """
    Demonstrate the complete folder scanning workflow
    """
    print("🚀 DEMO: Folder-Scanning Runbook Intelligence")
    print("=" * 60)
    
    # Initialize the scanner
    scanner = RunbookFolderScanner(
        knowledge_base_path="demo_knowledge.db",
        model_name="codellama:7b-instruct"
    )
    
    # Example 1: Scan SHFE runbook folder
    print("\n📁 Example 1: SHFE Runbook Folder")
    print("-" * 40)
    
    # This would be your actual folder path
    shfe_folder = "/runbooks/SHFEMR"  # Replace with actual path
    
    try:
        # Scan the folder and build knowledge
        knowledge = scanner.scan_runbook_folder(shfe_folder)
        
        print(f"✅ Knowledge Base Entry Created:")
        print(f"   • ID: {knowledge.runbook_id}")
        print(f"   • Dataset: {knowledge.dataset_name}")
        print(f"   • Domain: {knowledge.business_domain}")
        print(f"   • URL: {knowledge.primary_source_url}")
        print(f"   • Files: {len(knowledge.processed_files)}")
        print(f"   • Complexity: {knowledge.complexity_score:.2f}")
        print(f"   • Confidence: {knowledge.confidence_score:.2f}")
        print(f"   • Ready for Site Analysis: {knowledge.ready_for_site_analysis}")
        
        # Prepare for Site Intelligence
        if knowledge.ready_for_site_analysis:
            print(f"\n🎯 Preparing for Site Intelligence Engine...")
            site_prep = scanner.prepare_for_site_analysis(knowledge.runbook_id)
            
            if site_prep:
                print(f"✅ Site Analysis Package Ready:")
                print(f"   • Target URL: {site_prep.target_url}")
                print(f"   • Expected Data Patterns: {len(site_prep.expected_data_patterns)}")
                print(f"   • Navigation Hints: {len(site_prep.navigation_hints)}")
                print(f"   • Auth Required: {bool(site_prep.authentication_requirements)}")
                
                # This package is now ready to be passed to Site Intelligence Engine
                return site_prep
        
    except Exception as e:
        print(f"⚠️ Demo folder not found: {e}")
        return demo_with_sample_content(scanner)

def demo_with_sample_content(scanner: RunbookFolderScanner):
    """
    Demo with sample content when real folders aren't available
    """
    print("\n📋 Demo with Sample Content")
    print("-" * 40)
    
    # Create a temporary demo folder structure
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create sample runbook files
        
        # Main runbook file
        main_runbook = os.path.join(temp_dir, "SHFE_Runbook.txt")
        with open(main_runbook, 'w') as f:
            f.write("""
            SHFE Margin Ratio Scraper Runbook
            
            Dataset Name: SHFEMR
            Dataset Description: Shanghai Futures Exchange Margin Requirements
            Source: Shanghai Futures Exchange
            Link to source: https://www.shfe.com.cn/publicnotice/notice/
            Frequency: Weekdaily (BusinessDaily)
            SLA: Data published daily after 9AM EST
            
            INSTRUCTIONS:
            1. Navigate to https://www.shfe.com.cn/publicnotice/notice/
            2. Look for notices containing "margin ratio" adjustments
            3. Extract effective dates using pattern "from the closing settlement on [DATE]"
            4. Extract commodity names and their margin percentages
            5. Handle special cases like "restored to original levels"
            
            DATA TARGETS:
            - effective_date: Date when margin changes take effect (YYYY-MM-DD)
            - commodity: Commodity name (aluminum, zinc, gold, silver, etc.)
            - hedging_percentage: Margin ratio for hedging transactions
            - speculative_percentage: Margin ratio for speculative transactions
            
            EXTRACTION PATTERNS:
            - Look for: "trading margin ratio and price limit range will be adjusted as follows"
            - Date pattern: "from the closing settlement on ([^,]+)"
            - Multiple commodities: Handle sentences with multiple commodity names
            - Special cases: "restored to their original levels", "remains at"
            
            OUTPUT FORMAT:
            - XLS format with DATA and META files
            - ZIP archive containing both files
            - File naming: SHFEMR_DATA_YYYYMMDD.xls, SHFEMR_META_YYYYMMDD.xls
            """)
        
        # Supplementary instructions
        supplement_file = os.path.join(temp_dir, "extraction_examples.txt")
        with open(supplement_file, 'w') as f:
            f.write("""
            EXTRACTION EXAMPLES:
            
            Example 1 - Single commodity:
            "Starting from the closing settlement on May 29, 2025 (Thursday), 
            the trading margin ratio and price limit range will be adjusted as follows:
            The price limit of alumina futures contracts is adjusted to 9%, 
            the margin ratio for hedging transactions is adjusted to 10%, 
            and the margin ratio for speculative transactions is adjusted to 11%"
            
            Expected extraction:
            - effective_date: 2025-05-29
            - commodity: alumina
            - hedging_percentage: 10
            - speculative_percentage: 11
            
            Example 2 - Multiple commodities:
            "The price limits for aluminum, zinc, lead futures contracts were adjusted to 9%, 
            the margin ratio for hedging transactions was adjusted to 10%, 
            and the margin ratio for speculative transactions was adjusted to 11%"
            
            Expected extraction (3 separate entries):
            - aluminum: hedging=10%, speculative=11%
            - zinc: hedging=10%, speculative=11%  
            - lead: hedging=10%, speculative=11%
            """)
        
        # Sample screenshot reference
        screenshot_ref = os.path.join(temp_dir, "site_screenshot.txt")
        with open(screenshot_ref, 'w') as f:
            f.write("""
            Screenshot: SHFE Website Navigation
            
            Shows the main page at https://www.shfe.com.cn/publicnotice/notice/
            Key elements visible:
            - Navigation menu with "Public Notice" section
            - List of notices with dates and titles
            - Search functionality for finding margin-related notices
            - Pagination controls at bottom of page
            
            Navigation steps:
            1. Click on "Public Notice" in main menu
            2. Look for notices with titles containing "margin ratio" or "trading margin"
            3. Click on individual notices to view full content
            4. Extract margin adjustment information from notice text
            """)
        
        print(f"📂 Created demo folder: {temp_dir}")
        print(f"📄 Files created:")
        for file in os.listdir(temp_dir):
            print(f"   • {file}")
        
        # Now scan the demo folder
        knowledge = scanner.scan_runbook_folder(temp_dir)
        
        print(f"\n✅ Demo Knowledge Base Entry:")
        print(f"   • Dataset: {knowledge.dataset_name}")
        print(f"   • Domain: {knowledge.business_domain}")
        print(f"   • URL: {knowledge.primary_source_url}")
        print(f"   • Data Targets: {len(knowledge.data_targets)}")
        print(f"   • Files Processed: {knowledge.processed_files}")
        print(f"   • Ready for Site Analysis: {knowledge.ready_for_site_analysis}")
        
        # Show extracted data targets
        if knowledge.data_targets:
            print(f"\n📊 Extracted Data Targets:")
            for i, target in enumerate(knowledge.data_targets, 1):
                if isinstance(target, dict):
                    print(f"   {i}. {target.get('field_name', 'unknown')}: {target.get('description', 'no description')}")
        
        # Show extraction patterns
        if knowledge.extraction_patterns:
            print(f"\n🔍 Extraction Patterns:")
            patterns = knowledge.extraction_patterns
            if isinstance(patterns, dict):
                for key, value in patterns.items():
                    print(f"   • {key}: {str(value)[:100]}...")
        
        # Prepare for site analysis
        if knowledge.ready_for_site_analysis:
            print(f"\n🎯 Preparing Site Analysis Package...")
            site_prep = scanner.prepare_for_site_analysis(knowledge.runbook_id)
            
            if site_prep:
                print(f"✅ Ready for Site Intelligence Engine:")
                print(f"   • Target URL: {site_prep.target_url}")
                print(f"   • Business Context: {site_prep.business_context}")
                print(f"   • Navigation Hints: {site_prep.navigation_hints}")
                print(f"   • Special Instructions: {site_prep.special_instructions}")
                
                return site_prep
        
        return knowledge

def demo_knowledge_base_operations():
    """
    Demonstrate knowledge base operations
    """
    print("\n🗃️ DEMO: Knowledge Base Operations")
    print("-" * 40)
    
    scanner = RunbookFolderScanner(knowledge_base_path="demo_knowledge.db")
    
    # Get all runbooks
    all_runbooks = scanner.get_all_runbooks()
    print(f"📊 Total runbooks in knowledge base: {len(all_runbooks)}")
    
    for knowledge in all_runbooks:
        print(f"   • {knowledge.dataset_name} ({knowledge.business_domain})")
        print(f"     URL: {knowledge.primary_source_url}")
        print(f"     Ready: {knowledge.ready_for_site_analysis}")
        print(f"     Created: {knowledge.created_at}")
    
    # Search runbooks
    if all_runbooks:
        print(f"\n🔍 Searching for 'SHFE' runbooks:")
        shfe_runbooks = scanner.search_runbooks("SHFE")
        for knowledge in shfe_runbooks:
            print(f"   • Found: {knowledge.dataset_name}")
    
    return all_runbooks

def demo_site_analysis_handoff():
    """
    Demonstrate how to hand off to Site Intelligence Engine
    """
    print("\n🔄 DEMO: Handoff to Site Intelligence Engine")
    print("-" * 40)
    
    scanner = RunbookFolderScanner(knowledge_base_path="demo_knowledge.db")
    
    # Get runbooks ready for site analysis
    all_runbooks = scanner.get_all_runbooks()
    ready_runbooks = [r for r in all_runbooks if r.ready_for_site_analysis]
    
    print(f"🎯 Runbooks ready for Site Analysis: {len(ready_runbooks)}")
    
    for knowledge in ready_runbooks:
        print(f"\n📋 Processing: {knowledge.dataset_name}")
        
        # Prepare site analysis package
        site_prep = scanner.prepare_for_site_analysis(knowledge.runbook_id)
        
        if site_prep:
            print(f"✅ Site Analysis Package:")
            print(f"   • Target URL: {site_prep.target_url}")
            print(f"   • Expected Patterns: {len(site_prep.expected_data_patterns)}")
            print(f"   • Business Domain: {site_prep.business_context['domain']}")
            print(f"   • Complexity: {site_prep.business_context['complexity']}")
            
            # This is where you would call Site Intelligence Engine
            print(f"🔄 → Ready to call Site Intelligence Engine")
            print(f"     site_intelligence.analyze_site(site_prep)")
            
            return site_prep
    
    print("⚠️ No runbooks ready for site analysis")
    return None

# Main execution function
def main():
    """
    Main demonstration of the complete Runbook Intelligence workflow
    """
    print("🧠 RUNBOOK INTELLIGENCE ENGINE DEMO")
    print("=" * 60)
    print("This demo shows the complete workflow:")
    print("1. 📁 Folder scanning and file discovery")
    print("2. 🧠 LLM-powered requirement extraction") 
    print("3. 🗃️ Knowledge base building")
    print("4. 🎯 Site Analysis preparation")
    print("5. 🔄 Handoff to Site Intelligence Engine")
    print()
    
    try:
        # Step 1: Demo folder scanning
        site_prep = demo_folder_scanning()
        
        # Step 2: Demo knowledge base operations
        all_runbooks = demo_knowledge_base_operations()
        
        # Step 3: Demo site analysis handoff
        if not site_prep:
            site_prep = demo_site_analysis_handoff()
        
        print(f"\n🎉 DEMO COMPLETED SUCCESSFULLY!")
        print(f"📦 Site Analysis Package ready for Site Intelligence Engine")
        
        if site_prep:
            print(f"\n🔄 Next Step: Pass to Site Intelligence Engine")
            print(f"   site_intelligence_engine.analyze_site(site_prep)")
            
            # Show what Site Intelligence Engine will receive
            print(f"\n📋 Site Intelligence Engine will receive:")
            print(f"   • Target URL: {site_prep.target_url}")
            print(f"   • Expected Data: {[p.get('field_name') for p in site_prep.expected_data_patterns if isinstance(p, dict)]}")
            print(f"   • Navigation Hints: {site_prep.navigation_hints}")
            print(f"   • Success Criteria: {site_prep.success_criteria}")
        
        return site_prep
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()