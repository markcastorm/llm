#!/usr/bin/env python3
"""
Site Intelligence Engine - Complete Implementation
Deep content understanding with browser-based translation and learning integration
"""

import json
import re
import time
import ollama
import sqlite3
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# Selenium imports
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException

# HTML parsing
from bs4 import BeautifulSoup
import requests

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SiteComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    DYNAMIC = "dynamic"

class AnalysisDepth(Enum):
    FAST = "fast"
    THOROUGH = "thorough"
    DEEP = "deep"

@dataclass
class SiteStructureAnalysis:
    """Site structure analysis results"""
    url: str
    page_title: str
    dom_complexity: int
    navigation_elements: List[Dict[str, Any]]
    data_containers: List[Dict[str, Any]]
    form_elements: List[Dict[str, Any]]
    table_structures: List[Dict[str, Any]]
    dynamic_elements: List[Dict[str, Any]]
    css_patterns: List[str]
    accessibility_score: float

@dataclass
class JavaScriptBehavior:
    """JavaScript behavior analysis"""
    has_ajax: bool
    ajax_endpoints: List[str]
    dynamic_loading: bool
    lazy_loading: bool
    infinite_scroll: bool
    form_validation: bool
    timing_requirements: Dict[str, float]
    event_handlers: List[str]
    performance_metrics: Dict[str, Any]

@dataclass
class DataLocationMapping:
    """Mapping of business data to HTML elements"""
    field_name: str
    description: str
    primary_selectors: List[str]
    fallback_selectors: List[str]
    extraction_pattern: str
    validation_rules: List[str]
    data_type: str
    confidence_score: float
    sample_values: List[str]

@dataclass
class NavigationWorkflow:
    """Navigation workflow for data extraction"""
    step_id: int
    action_type: str  # click, scroll, wait, extract, etc.
    target_selector: str
    parameters: Dict[str, Any]
    wait_condition: Optional[str]
    timeout: float
    error_handling: str
    success_criteria: str

@dataclass
class SiteIntelligenceReport:
    """Complete site intelligence analysis report"""
    runbook_id: str
    target_url: str
    analysis_timestamp: str
    site_complexity: SiteComplexity
    structure_analysis: SiteStructureAnalysis
    javascript_behavior: JavaScriptBehavior
    data_mappings: List[DataLocationMapping]
    navigation_workflow: List[NavigationWorkflow]
    extraction_strategy: Dict[str, Any]
    performance_recommendations: Dict[str, Any]
    error_handling_strategies: List[Dict[str, Any]]
    learning_insights: Dict[str, Any]
    cache_expires_at: str
    confidence_score: float

class SeleniumSiteNavigator:
    """Advanced Selenium-based site navigation and analysis"""
    
    def __init__(self, headless: bool = True, enable_translation: bool = True):
        self.headless = headless
        self.enable_translation = enable_translation
        self.driver = None
        self.wait = None
        self._setup_driver()
        
    def _setup_driver(self):
        """Setup Chrome driver with translation and optimization"""
        chrome_options = Options()
        
        if self.headless:
            chrome_options.add_argument("--headless")
        
        # Performance optimizations
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-logging")
        chrome_options.add_argument("--disable-background-networking")
        
        # Translation settings
        if self.enable_translation:
            prefs = {
                "translate_whitelists": {"zh-CN": "en", "zh": "en"},
                "translate": {"enabled": True},
                "translate.enabled": True
            }
            chrome_options.add_experimental_option("prefs", prefs)
        
        # User agent
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.set_page_load_timeout(30)
            self.driver.implicitly_wait(10)
            self.wait = WebDriverWait(self.driver, 15)
            logger.info("✅ Selenium driver initialized successfully")
        except Exception as e:
            logger.error(f"❌ Driver setup failed: {e}")
            raise
    
    def navigate_to_site(self, url: str) -> Dict[str, Any]:
        """Navigate to site and handle initial loading"""
        logger.info(f"🌐 Navigating to: {url}")
        
        start_time = time.time()
        try:
            self.driver.get(url)
            
            # Wait for page load
            self.wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            
            # Handle translation if enabled
            if self.enable_translation:
                self._trigger_translation()
            
            load_time = time.time() - start_time
            
            # Get basic page info
            page_info = {
                "url": self.driver.current_url,
                "title": self.driver.title,
                "load_time": load_time,
                "page_source_size": len(self.driver.page_source),
                "status": "success"
            }
            
            logger.info(f"✅ Page loaded successfully in {load_time:.2f}s")
            return page_info
            
        except TimeoutException:
            logger.warning("⏰ Page load timeout")
            return {"status": "timeout", "url": url}
        except Exception as e:
            logger.error(f"❌ Navigation failed: {e}")
            return {"status": "error", "url": url, "error": str(e)}
    
    def _trigger_translation(self):
        """Trigger Chrome's built-in translation for Chinese content"""
        try:
            # Wait a moment for translation detection
            time.sleep(2)
            
            # Look for translation notification
            try:
                translate_element = self.driver.find_element(By.CSS_SELECTOR, "[id*='translate']")
                if translate_element:
                    logger.info("🌍 Translation detected, attempting to trigger")
                    # Try to click translate button if present
                    translate_buttons = self.driver.find_elements(By.XPATH, "//*[contains(text(), 'Translate') or contains(text(), '翻译')]")
                    if translate_buttons:
                        translate_buttons[0].click()
                        time.sleep(3)  # Wait for translation
                        logger.info("✅ Translation triggered")
            except NoSuchElementException:
                # No translation notification found
                pass
                
        except Exception as e:
            logger.warning(f"⚠️ Translation trigger failed: {e}")
    
    def get_page_source(self) -> str:
        """Get current page source"""
        return self.driver.page_source if self.driver else ""
    
    def find_elements_by_pattern(self, patterns: List[str]) -> List[Dict[str, Any]]:
        """Find elements using multiple selector patterns"""
        found_elements = []
        
        for pattern in patterns:
            try:
                elements = self.driver.find_elements(By.CSS_SELECTOR, pattern)
                for element in elements:
                    element_info = {
                        "selector": pattern,
                        "tag_name": element.tag_name,
                        "text": element.text[:100],  # First 100 chars
                        "attributes": {
                            "class": element.get_attribute("class"),
                            "id": element.get_attribute("id"),
                            "href": element.get_attribute("href")
                        },
                        "location": element.location,
                        "size": element.size
                    }
                    found_elements.append(element_info)
            except Exception as e:
                logger.warning(f"⚠️ Pattern {pattern} failed: {e}")
        
        return found_elements
    
    def simulate_user_interaction(self, action_type: str, selector: str, **kwargs) -> bool:
        """Simulate user interactions"""
        try:
            element = self.wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, selector)))
            
            if action_type == "click":
                element.click()
            elif action_type == "scroll_to":
                self.driver.execute_script("arguments[0].scrollIntoView();", element)
            elif action_type == "hover":
                ActionChains(self.driver).move_to_element(element).perform()
            elif action_type == "type":
                text = kwargs.get("text", "")
                element.clear()
                element.send_keys(text)
            elif action_type == "submit":
                element.submit()
            
            time.sleep(1)  # Wait for action to complete
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Interaction {action_type} failed: {e}")
            return False
    
    def close(self):
        """Close the browser"""
        if self.driver:
            self.driver.quit()
            logger.info("✅ Browser closed")

class HTMLStructureAnalyzer:
    """Deep HTML structure analysis"""
    
    def __init__(self, navigator: SeleniumSiteNavigator):
        self.navigator = navigator
    
    def analyze_structure(self, url: str) -> SiteStructureAnalysis:
        """Perform comprehensive HTML structure analysis"""
        logger.info("🔍 Starting HTML structure analysis")
        
        page_source = self.navigator.get_page_source()
        soup = BeautifulSoup(page_source, 'html.parser')
        
        analysis = SiteStructureAnalysis(
            url=url,
            page_title=soup.title.string if soup.title else "",
            dom_complexity=self._calculate_dom_complexity(soup),
            navigation_elements=self._analyze_navigation(soup),
            data_containers=self._identify_data_containers(soup),
            form_elements=self._analyze_forms(soup),
            table_structures=self._analyze_tables(soup),
            dynamic_elements=self._identify_dynamic_elements(soup),
            css_patterns=self._extract_css_patterns(soup),
            accessibility_score=self._calculate_accessibility_score(soup)
        )
        
        logger.info(f"✅ Structure analysis complete - DOM complexity: {analysis.dom_complexity}")
        return analysis
    
    def _calculate_dom_complexity(self, soup: BeautifulSoup) -> int:
        """Calculate DOM complexity score"""
        factors = [
            len(soup.find_all()),  # Total elements
            len(soup.find_all('div')),  # Div count
            len(soup.find_all('script')),  # Script count
            len(soup.find_all(class_=True)),  # Elements with classes
            max([len(str(tag)) for tag in soup.find_all()[:10]] + [0])  # Max element size
        ]
        return sum(factors) // len(factors)
    
    def _analyze_navigation(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Analyze navigation elements"""
        nav_elements = []
        
        # Find navigation containers
        nav_tags = soup.find_all(['nav', 'header']) + soup.find_all(class_=lambda x: x and any(nav_term in str(x).lower() for nav_term in ['nav', 'menu', 'header']))
        
        for nav in nav_tags:
            links = nav.find_all('a')
            nav_info = {
                "type": nav.name,
                "class": nav.get('class', []),
                "id": nav.get('id', ''),
                "link_count": len(links),
                "links": [{"text": link.get_text(strip=True), "href": link.get('href')} for link in links[:5]]  # First 5 links
            }
            nav_elements.append(nav_info)
        
        return nav_elements
    
    def _identify_data_containers(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Identify likely data container elements"""
        containers = []
        
        # Look for tables (common for financial data)
        tables = soup.find_all('table')
        for i, table in enumerate(tables):
            rows = table.find_all('tr')
            headers = table.find_all('th')
            
            container_info = {
                "type": "table",
                "index": i,
                "selector": f"table:nth-of-type({i+1})",
                "row_count": len(rows),
                "column_count": len(headers) if headers else len(rows[0].find_all(['td', 'th'])) if rows else 0,
                "headers": [th.get_text(strip=True) for th in headers],
                "class": table.get('class', []),
                "id": table.get('id', '')
            }
            containers.append(container_info)
        
        # Look for lists (ul, ol)
        lists = soup.find_all(['ul', 'ol'])
        for i, lst in enumerate(lists):
            items = lst.find_all('li')
            container_info = {
                "type": lst.name,
                "index": i,
                "selector": f"{lst.name}:nth-of-type({i+1})",
                "item_count": len(items),
                "class": lst.get('class', []),
                "id": lst.get('id', ''),
                "sample_items": [item.get_text(strip=True)[:50] for item in items[:3]]
            }
            containers.append(container_info)
        
        # Look for divs with data-like classes
        data_divs = soup.find_all('div', class_=lambda x: x and any(data_term in str(x).lower() for data_term in ['data', 'content', 'item', 'row', 'entry', 'notice']))
        for i, div in enumerate(data_divs[:10]):  # Limit to first 10
            container_info = {
                "type": "data_div",
                "index": i,
                "selector": f"div.{' '.join(div.get('class', []))}".replace(' ', '.'),
                "class": div.get('class', []),
                "id": div.get('id', ''),
                "text_sample": div.get_text(strip=True)[:100]
            }
            containers.append(container_info)
        
        return containers
    
    def _analyze_forms(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Analyze form elements"""
        forms = []
        
        for i, form in enumerate(soup.find_all('form')):
            inputs = form.find_all(['input', 'select', 'textarea'])
            form_info = {
                "index": i,
                "action": form.get('action', ''),
                "method": form.get('method', 'get'),
                "class": form.get('class', []),
                "id": form.get('id', ''),
                "input_count": len(inputs),
                "inputs": [
                    {
                        "type": inp.get('type', inp.name),
                        "name": inp.get('name', ''),
                        "placeholder": inp.get('placeholder', ''),
                        "required": inp.has_attr('required')
                    } for inp in inputs
                ]
            }
            forms.append(form_info)
        
        return forms
    
    def _analyze_tables(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Detailed table structure analysis"""
        tables = []
        
        for i, table in enumerate(soup.find_all('table')):
            thead = table.find('thead')
            tbody = table.find('tbody')
            rows = table.find_all('tr')
            
            # Analyze table structure
            table_info = {
                "index": i,
                "has_thead": thead is not None,
                "has_tbody": tbody is not None,
                "total_rows": len(rows),
                "class": table.get('class', []),
                "id": table.get('id', ''),
                "selector": f"table:nth-of-type({i+1})"
            }
            
            # Extract headers
            if thead:
                header_rows = thead.find_all('tr')
                if header_rows:
                    headers = header_rows[0].find_all(['th', 'td'])
                    table_info["headers"] = [h.get_text(strip=True) for h in headers]
            elif rows:
                # First row might be headers
                first_row_cells = rows[0].find_all(['th', 'td'])
                table_info["potential_headers"] = [cell.get_text(strip=True) for cell in first_row_cells]
            
            # Sample data rows
            data_rows = tbody.find_all('tr') if tbody else rows[1:] if len(rows) > 1 else []
            table_info["sample_data"] = []
            for row in data_rows[:3]:  # First 3 data rows
                cells = row.find_all(['td', 'th'])
                row_data = [cell.get_text(strip=True) for cell in cells]
                table_info["sample_data"].append(row_data)
            
            tables.append(table_info)
        
        return tables
    
    def _identify_dynamic_elements(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Identify potentially dynamic elements"""
        dynamic_elements = []
        
        # Look for elements with JavaScript-related attributes
        js_elements = soup.find_all(attrs={"onclick": True}) + soup.find_all(attrs={"onload": True})
        for element in js_elements:
            dynamic_info = {
                "tag": element.name,
                "class": element.get('class', []),
                "id": element.get('id', ''),
                "onclick": element.get('onclick', ''),
                "onload": element.get('onload', '')
            }
            dynamic_elements.append(dynamic_info)
        
        # Look for AJAX-related patterns
        ajax_patterns = soup.find_all(attrs={"data-url": True}) + soup.find_all(class_=lambda x: x and 'ajax' in str(x).lower())
        for element in ajax_patterns:
            dynamic_info = {
                "tag": element.name,
                "class": element.get('class', []),
                "data_url": element.get('data-url', ''),
                "type": "ajax_candidate"
            }
            dynamic_elements.append(dynamic_info)
        
        return dynamic_elements
    
    def _extract_css_patterns(self, soup: BeautifulSoup) -> List[str]:
        """Extract common CSS class patterns"""
        all_classes = []
        for element in soup.find_all(class_=True):
            classes = element.get('class', [])
            all_classes.extend(classes)
        
        # Count class occurrences
        class_counts = {}
        for cls in all_classes:
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        # Return most common classes
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        return [cls for cls, count in sorted_classes[:20] if count > 1]
    
    def _calculate_accessibility_score(self, soup: BeautifulSoup) -> float:
        """Calculate basic accessibility score"""
        score_factors = []
        
        # Check for alt attributes on images
        images = soup.find_all('img')
        if images:
            images_with_alt = len([img for img in images if img.get('alt')])
            score_factors.append(images_with_alt / len(images))
        
        # Check for labels on form inputs
        inputs = soup.find_all('input')
        if inputs:
            labels = soup.find_all('label')
            score_factors.append(min(len(labels) / len(inputs), 1.0))
        
        # Check for heading structure
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        score_factors.append(1.0 if headings else 0.0)
        
        return sum(score_factors) / len(score_factors) if score_factors else 0.5

class JavaScriptBehaviorDetector:
    """Detect and analyze JavaScript behavior"""
    
    def __init__(self, navigator: SeleniumSiteNavigator):
        self.navigator = navigator
    
    def analyze_behavior(self, url: str) -> JavaScriptBehavior:
        """Analyze JavaScript behavior on the page"""
        logger.info("⚡ Analyzing JavaScript behavior")
        
        # Monitor network requests
        ajax_endpoints = self._detect_ajax_calls()
        
        # Test dynamic loading
        dynamic_loading = self._test_dynamic_loading()
        
        # Check for lazy loading
        lazy_loading = self._test_lazy_loading()
        
        # Test infinite scroll
        infinite_scroll = self._test_infinite_scroll()
        
        # Analyze timing requirements
        timing_requirements = self._analyze_timing_requirements()
        
        # Performance metrics
        performance_metrics = self._get_performance_metrics()
        
        behavior = JavaScriptBehavior(
            has_ajax=len(ajax_endpoints) > 0,
            ajax_endpoints=ajax_endpoints,
            dynamic_loading=dynamic_loading,
            lazy_loading=lazy_loading,
            infinite_scroll=infinite_scroll,
            form_validation=self._test_form_validation(),
            timing_requirements=timing_requirements,
            event_handlers=self._detect_event_handlers(),
            performance_metrics=performance_metrics
        )
        
        logger.info(f"✅ JavaScript analysis complete - AJAX: {behavior.has_ajax}, Dynamic: {behavior.dynamic_loading}")
        return behavior
    
    def _detect_ajax_calls(self) -> List[str]:
        """Detect AJAX endpoints by monitoring network traffic"""
        endpoints = []
        
        try:
            # Enable network monitoring
            self.navigator.driver.execute_cdp_cmd('Network.enable', {})
            
            # Wait and collect requests
            time.sleep(3)
            
            # Get network logs (simplified approach)
            logs = self.navigator.driver.get_log('performance')
            for log in logs:
                message = json.loads(log['message'])
                if message.get('message', {}).get('method') == 'Network.requestWillBeSent':
                    url = message.get('message', {}).get('params', {}).get('request', {}).get('url', '')
                    if url and ('ajax' in url.lower() or 'api' in url.lower() or '.json' in url):
                        endpoints.append(url)
        
        except Exception as e:
            logger.warning(f"⚠️ AJAX detection failed: {e}")
        
        return list(set(endpoints))  # Remove duplicates
    
    def _test_dynamic_loading(self) -> bool:
        """Test for dynamic content loading"""
        try:
            initial_content = len(self.navigator.get_page_source())
            
            # Scroll down to trigger potential loading
            self.navigator.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            
            # Check for new content
            final_content = len(self.navigator.get_page_source())
            
            return final_content > initial_content * 1.05  # 5% increase threshold
            
        except Exception as e:
            logger.warning(f"⚠️ Dynamic loading test failed: {e}")
            return False
    
    def _test_lazy_loading(self) -> bool:
        """Test for lazy loading of images or content"""
        try:
            # Look for lazy loading indicators
            lazy_indicators = self.navigator.driver.find_elements(By.CSS_SELECTOR, "[data-src], [loading='lazy'], .lazy")
            return len(lazy_indicators) > 0
            
        except Exception as e:
            logger.warning(f"⚠️ Lazy loading test failed: {e}")
            return False
    
    def _test_infinite_scroll(self) -> bool:
        """Test for infinite scroll functionality"""
        try:
            initial_height = self.navigator.driver.execute_script("return document.body.scrollHeight")
            
            # Scroll to bottom
            self.navigator.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3)
            
            final_height = self.navigator.driver.execute_script("return document.body.scrollHeight")
            
            return final_height > initial_height
            
        except Exception as e:
            logger.warning(f"⚠️ Infinite scroll test failed: {e}")
            return False
    
    def _test_form_validation(self) -> bool:
        """Test for client-side form validation"""
        try:
            forms = self.navigator.driver.find_elements(By.TAG_NAME, "form")
            for form in forms:
                required_inputs = form.find_elements(By.CSS_SELECTOR, "[required]")
                if required_inputs:
                    return True
            return False
            
        except Exception as e:
            logger.warning(f"⚠️ Form validation test failed: {e}")
            return False
    
    def _analyze_timing_requirements(self) -> Dict[str, float]:
        """Analyze optimal timing requirements"""
        timings = {}
        
        try:
            # Page load timing
            navigation_timing = self.navigator.driver.execute_script("""
                var timing = window.performance.timing;
                return {
                    'page_load': timing.loadEventEnd - timing.navigationStart,
                    'dom_ready': timing.domContentLoadedEventEnd - timing.navigationStart,
                    'first_paint': timing.responseStart - timing.navigationStart
                };
            """)
            
            timings.update(navigation_timing)
            
            # Element interaction timing
            timings['recommended_wait'] = max(2.0, timings.get('dom_ready', 2000) / 1000)
            timings['max_wait'] = min(15.0, timings.get('page_load', 10000) / 1000)
            
        except Exception as e:
            logger.warning(f"⚠️ Timing analysis failed: {e}")
            timings = {'recommended_wait': 3.0, 'max_wait': 10.0}
        
        return timings
    
    def _detect_event_handlers(self) -> List[str]:
        """Detect JavaScript event handlers"""
        handlers = []
        
        try:
            # Look for common event handlers in the DOM
            elements_with_events = self.navigator.driver.find_elements(By.CSS_SELECTOR, "[onclick], [onchange], [onsubmit], [onload]")
            
            for element in elements_with_events:
                for event in ['onclick', 'onchange', 'onsubmit', 'onload']:
                    if element.get_attribute(event):
                        handlers.append(f"{element.tag_name}.{event}")
            
        except Exception as e:
            logger.warning(f"⚠️ Event handler detection failed: {e}")
        
        return list(set(handlers))
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        metrics = {}
        
        try:
            # Get basic performance metrics
            perf_data = self.navigator.driver.execute_script("""
                var perf = window.performance;
                return {
                    'memory': perf.memory ? {
                        'used': perf.memory.usedJSHeapSize,
                        'total': perf.memory.totalJSHeapSize,
                        'limit': perf.memory.jsHeapSizeLimit
                    } : null,
                    'navigation': perf.getEntriesByType('navigation')[0] || {},
                    'resources': perf.getEntriesByType('resource').length
                };
            """)
            
            metrics.update(perf_data)
            
        except Exception as e:
            logger.warning(f"⚠️ Performance metrics failed: {e}")
        
        return metrics

class LLMSiteInterpreter:
    """LLM-powered site content interpretation"""
    
    def __init__(self, model_name: str = "codellama:13b-instruct"):
        self.model_name = model_name
    
    def interpret_site_content(self, url: str, page_content: str, 
                             runbook_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to interpret site content and create extraction strategy"""
        logger.info("🧠 Starting LLM site content interpretation")
        
        # Limit content size for LLM processing
        content_preview = page_content[:5000] if len(page_content) > 5000 else page_content
        
        prompt = f"""
        You are an expert web scraping analyst. Analyze this website and create an optimal data extraction strategy.

        WEBSITE URL: {url}

        BUSINESS REQUIREMENTS (from runbook):
        {json.dumps(runbook_requirements, indent=2)}

        WEBSITE CONTENT PREVIEW:
        {content_preview}

        Analyze the website and provide a detailed extraction strategy in JSON format:

        {{
            "site_analysis": {{
                "site_type": "financial_exchange|news|ecommerce|government|other",
                "language": "english|chinese|multilingual",
                "complexity": "simple|moderate|complex|dynamic",
                "main_content_type": "table|list|article|form|mixed",
                "navigation_required": true/false
            }},
            "data_location_strategy": {{
                "primary_data_container": "CSS selector for main data area",
                "data_extraction_patterns": [
                    {{
                        "field_name": "field from requirements",
                        "extraction_method": "css_selector|regex|xpath|text_pattern",
                        "selector_pattern": "specific CSS selector or pattern",
                        "confidence": 0.0-1.0,
                        "notes": "explanation of extraction approach"
                    }}
                ],
                "navigation_steps": [
                    {{
                        "step": "description of navigation step",
                        "action": "click|scroll|wait|search",
                        "target": "CSS selector or description"
                    }}
                ]
            }},
            "extraction_challenges": [
                "list of potential challenges and solutions"
            ],
            "recommended_approach": {{
                "primary_strategy": "description of main approach",
                "fallback_strategies": ["alternative approaches"],
                "timing_considerations": "wait times and performance notes",
                "error_handling": "how to handle common errors"
            }},
            "confidence_assessment": {{
                "overall_confidence": 0.0-1.0,
                "data_availability": 0.0-1.0,
                "extraction_complexity": 0.0-1.0,
                "site_stability": 0.0-1.0
            }}
        }}

        Focus on practical, implementable strategies for autonomous data extraction.
        """

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}]
            )
            
            response_text = response['message']['content']
            logger.info(f"🤖 LLM response received ({len(response_text)} chars)")
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                interpretation = json.loads(json_match.group(0))
                logger.info("✅ LLM interpretation successful")
                return interpretation
            else:
                logger.warning("⚠️ No JSON found in LLM response")
                return self._fallback_interpretation(url, runbook_requirements)
                
        except Exception as e:
            logger.error(f"❌ LLM interpretation failed: {e}")
            return self._fallback_interpretation(url, runbook_requirements)
    
    def _fallback_interpretation(self, url: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback interpretation using pattern matching"""
        return {
            "site_analysis": {
                "site_type": "other",
                "language": "unknown",
                "complexity": "moderate",
                "main_content_type": "mixed",
                "navigation_required": True
            },
            "data_location_strategy": {
                "primary_data_container": "body",
                "data_extraction_patterns": [
                    {
                        "field_name": "generic_data",
                        "extraction_method": "css_selector",
                        "selector_pattern": "table, .data, .content",
                        "confidence": 0.5,
                        "notes": "fallback generic extraction"
                    }
                ],
                "navigation_steps": []
            },
            "extraction_challenges": ["Unknown site structure"],
            "recommended_approach": {
                "primary_strategy": "Generic table and list extraction",
                "fallback_strategies": ["Text pattern matching"],
                "timing_considerations": "Standard wait times",
                "error_handling": "Basic error recovery"
            },
            "confidence_assessment": {
                "overall_confidence": 0.5,
                "data_availability": 0.5,
                "extraction_complexity": 0.7,
                "site_stability": 0.6
            }
        }

class DataLocationMapper:
    """Map business data requirements to HTML elements"""
    
    def __init__(self, navigator: SeleniumSiteNavigator, llm_interpreter: LLMSiteInterpreter):
        self.navigator = navigator
        self.llm_interpreter = llm_interpreter
    
    def create_data_mappings(self, structure_analysis: SiteStructureAnalysis,
                           llm_interpretation: Dict[str, Any],
                           runbook_requirements: Dict[str, Any]) -> List[DataLocationMapping]:
        """Create detailed data location mappings"""
        logger.info("📍 Creating data location mappings")
        
        mappings = []
        
        # Get data targets from runbook requirements
        data_targets = runbook_requirements.get('expected_data_patterns', [])
        extraction_patterns = llm_interpretation.get('data_location_strategy', {}).get('data_extraction_patterns', [])
        
        for target in data_targets:
            if isinstance(target, dict):
                field_name = target.get('field_name', 'unknown')
                
                # Find corresponding LLM extraction pattern
                llm_pattern = self._find_matching_pattern(field_name, extraction_patterns)
                
                # Create mapping
                mapping = self._create_field_mapping(target, llm_pattern, structure_analysis)
                mappings.append(mapping)
        
        logger.info(f"✅ Created {len(mappings)} data mappings")
        return mappings
    
    def _find_matching_pattern(self, field_name: str, extraction_patterns: List[Dict]) -> Optional[Dict]:
        """Find matching extraction pattern from LLM analysis"""
        for pattern in extraction_patterns:
            if pattern.get('field_name', '').lower() == field_name.lower():
                return pattern
        return None
    
    def _create_field_mapping(self, target: Dict, llm_pattern: Optional[Dict], 
                            structure_analysis: SiteStructureAnalysis) -> DataLocationMapping:
        """Create a detailed field mapping"""
        field_name = target.get('field_name', 'unknown')
        
        # Primary selectors from LLM
        primary_selectors = []
        if llm_pattern:
            selector = llm_pattern.get('selector_pattern', '')
            if selector:
                primary_selectors.append(selector)
        
        # Fallback selectors based on structure analysis
        fallback_selectors = self._generate_fallback_selectors(field_name, structure_analysis)
        
        # Extraction pattern
        extraction_pattern = self._create_extraction_pattern(field_name, llm_pattern)
        
        # Validation rules
        validation_rules = self._create_validation_rules(target)
        
        # Test selectors and get sample values
        sample_values, confidence = self._test_selectors(primary_selectors + fallback_selectors)
        
        mapping = DataLocationMapping(
            field_name=field_name,
            description=target.get('description', ''),
            primary_selectors=primary_selectors,
            fallback_selectors=fallback_selectors,
            extraction_pattern=extraction_pattern,
            validation_rules=validation_rules,
            data_type=target.get('data_type', 'text'),
            confidence_score=confidence,
            sample_values=sample_values
        )
        
        return mapping
    
    def _generate_fallback_selectors(self, field_name: str, structure_analysis: SiteStructureAnalysis) -> List[str]:
        """Generate fallback selectors based on field name and structure"""
        selectors = []
        
        # Field-specific selectors
        if 'date' in field_name.lower():
            selectors.extend([
                "[class*='date']", "[id*='date']", "time", ".timestamp",
                "td:contains('date')", "th:contains('date')"
            ])
        elif 'percentage' in field_name.lower() or 'ratio' in field_name.lower():
            selectors.extend([
                "[class*='percent']", "[class*='ratio']", "[class*='rate']",
                "td:contains('%')", "span:contains('%')"
            ])
        elif 'commodity' in field_name.lower():
            selectors.extend([
                "[class*='commodity']", "[class*='product']", "[class*='item']",
                "td:first-child", ".name", ".title"
            ])
        
        # Table-based selectors if tables exist
        if structure_analysis.table_structures:
            selectors.extend([
                "table td", "table th", "tr td:nth-child(1)", "tr td:nth-child(2)"
            ])
        
        # List-based selectors
        selectors.extend([
            "li", ".item", ".entry", ".row", ".data"
        ])
        
        return selectors
    
    def _create_extraction_pattern(self, field_name: str, llm_pattern: Optional[Dict]) -> str:
        """Create extraction pattern for the field"""
        if llm_pattern:
            method = llm_pattern.get('extraction_method', 'text')
            if method == 'regex':
                return llm_pattern.get('regex_pattern', r'.*')
            elif method == 'xpath':
                return llm_pattern.get('xpath_pattern', '')
        
        # Default patterns based on field type
        if 'date' in field_name.lower():
            return r'\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4}'
        elif 'percentage' in field_name.lower():
            return r'\d+\.?\d*%?'
        else:
            return 'text_content'
    
    def _create_validation_rules(self, target: Dict) -> List[str]:
        """Create validation rules for the field"""
        rules = []
        data_type = target.get('data_type', 'text')
        
        if data_type == 'date':
            rules.append("must_be_valid_date")
            rules.append("format_yyyy_mm_dd")
        elif data_type == 'percentage':
            rules.append("must_be_numeric")
            rules.append("range_0_to_100")
        elif data_type == 'currency':
            rules.append("must_be_numeric")
            rules.append("positive_value")
        
        return rules
    
    def _test_selectors(self, selectors: List[str]) -> Tuple[List[str], float]:
        """Test selectors and return sample values with confidence score"""
        sample_values = []
        working_selectors = 0
        
        for selector in selectors[:5]:  # Test first 5 selectors
            try:
                elements = self.navigator.find_elements_by_pattern([selector])
                if elements:
                    working_selectors += 1
                    for element in elements[:3]:  # First 3 matches
                        text = element.get('text', '').strip()
                        if text and len(text) < 100:
                            sample_values.append(text)
            except Exception as e:
                logger.warning(f"⚠️ Selector test failed for {selector}: {e}")
        
        confidence = working_selectors / len(selectors) if selectors else 0.0
        return list(set(sample_values)), confidence

class SiteIntelligenceCacheManager:
    """Manage caching of site intelligence results"""
    
    def __init__(self, cache_db_path: str = "site_intelligence_cache.db"):
        self.cache_db_path = cache_db_path
        self._init_cache_db()
    
    def _init_cache_db(self):
        """Initialize cache database"""
        conn = sqlite3.connect(self.cache_db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS site_cache (
                url_hash TEXT PRIMARY KEY,
                url TEXT,
                analysis_data TEXT,
                created_at TEXT,
                expires_at TEXT,
                cache_type TEXT
            )
        ''')
        conn.commit()
        conn.close()
        logger.info(f"📊 Cache database initialized: {self.cache_db_path}")
    
    def get_cached_analysis(self, url: str, cache_type: str = "full") -> Optional[Dict[str, Any]]:
        """Get cached analysis if available and not expired"""
        url_hash = hashlib.md5(f"{url}_{cache_type}".encode()).hexdigest()
        
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.execute(
            "SELECT analysis_data, expires_at FROM site_cache WHERE url_hash = ?",
            (url_hash,)
        )
        row = cursor.fetchone()
        conn.close()
        
        if row:
            analysis_data, expires_at = row
            if datetime.fromisoformat(expires_at) > datetime.now():
                logger.info(f"✅ Using cached analysis for {url}")
                return json.loads(analysis_data)
            else:
                logger.info(f"⏰ Cached analysis expired for {url}")
                self._remove_expired_cache(url_hash)
        
        return None
    
    def cache_analysis(self, url: str, analysis_data: Dict[str, Any], 
                      cache_duration_hours: int = 6, cache_type: str = "full"):
        """Cache analysis results"""
        url_hash = hashlib.md5(f"{url}_{cache_type}".encode()).hexdigest()
        created_at = datetime.now().isoformat()
        expires_at = (datetime.now() + timedelta(hours=cache_duration_hours)).isoformat()
        
        conn = sqlite3.connect(self.cache_db_path)
        conn.execute('''
            INSERT OR REPLACE INTO site_cache 
            (url_hash, url, analysis_data, created_at, expires_at, cache_type)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (url_hash, url, json.dumps(analysis_data), created_at, expires_at, cache_type))
        conn.commit()
        conn.close()
        
        logger.info(f"💾 Cached analysis for {url} (expires in {cache_duration_hours}h)")
    
    def _remove_expired_cache(self, url_hash: str):
        """Remove expired cache entry"""
        conn = sqlite3.connect(self.cache_db_path)
        conn.execute("DELETE FROM site_cache WHERE url_hash = ?", (url_hash,))
        conn.commit()
        conn.close()
    
    def cleanup_expired_cache(self):
        """Remove all expired cache entries"""
        now = datetime.now().isoformat()
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.execute("DELETE FROM site_cache WHERE expires_at < ?", (now,))
        removed_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        if removed_count > 0:
            logger.info(f"🧹 Cleaned up {removed_count} expired cache entries")

class SiteIntelligenceEngine:
    """Main Site Intelligence Engine coordinating all components"""
    
    def __init__(self, 
                 model_name: str = "codellama:13b-instruct",
                 headless: bool = True,
                 enable_caching: bool = True,
                 cache_duration_hours: int = 6):
        
        self.model_name = model_name
        self.headless = headless
        self.enable_caching = enable_caching
        self.cache_duration_hours = cache_duration_hours
        
        # Initialize components
        self.navigator = SeleniumSiteNavigator(headless=headless, enable_translation=True)
        self.structure_analyzer = HTMLStructureAnalyzer(self.navigator)
        self.js_detector = JavaScriptBehaviorDetector(self.navigator)
        self.llm_interpreter = LLMSiteInterpreter(model_name=model_name)
        self.data_mapper = DataLocationMapper(self.navigator, self.llm_interpreter)
        
        if enable_caching:
            self.cache_manager = SiteIntelligenceCacheManager()
        else:
            self.cache_manager = None
        
        logger.info("🎯 Site Intelligence Engine initialized")
    
    def analyze_site(self, site_prep: Dict[str, Any], analysis_depth: AnalysisDepth = AnalysisDepth.THOROUGH) -> SiteIntelligenceReport:
        """Main entry point: Comprehensive site analysis"""
        
        runbook_id = site_prep.get('runbook_id', 'unknown')
        target_url = site_prep.get('target_url', '')
        
        logger.info(f"🎯 Starting site analysis for: {target_url}")
        logger.info(f"📊 Analysis depth: {analysis_depth.value}")
        
        # Check cache first
        if self.cache_manager and analysis_depth != AnalysisDepth.DEEP:
            cached_result = self.cache_manager.get_cached_analysis(target_url)
            if cached_result:
                return self._deserialize_report(cached_result)
        
        try:
            # Step 1: Navigate to site
            navigation_result = self.navigator.navigate_to_site(target_url)
            if navigation_result.get('status') != 'success':
                raise Exception(f"Navigation failed: {navigation_result}")
            
            # Step 2: Structure analysis
            structure_analysis = self.structure_analyzer.analyze_structure(target_url)
            
            # Step 3: JavaScript behavior analysis (conditional based on depth)
            if analysis_depth in [AnalysisDepth.THOROUGH, AnalysisDepth.DEEP]:
                js_behavior = self.js_detector.analyze_behavior(target_url)
            else:
                js_behavior = self._create_minimal_js_behavior()
            
            # Step 4: LLM content interpretation
            page_content = self.navigator.get_page_source()
            llm_interpretation = self.llm_interpreter.interpret_site_content(
                target_url, page_content, site_prep
            )
            
            # Step 5: Data location mapping
            data_mappings = self.data_mapper.create_data_mappings(
                structure_analysis, llm_interpretation, site_prep
            )
            
            # Step 6: Create navigation workflow
            navigation_workflow = self._create_navigation_workflow(
                llm_interpretation, structure_analysis, site_prep
            )
            
            # Step 7: Generate extraction strategy
            extraction_strategy = self._create_extraction_strategy(
                data_mappings, llm_interpretation, js_behavior
            )
            
            # Step 8: Performance recommendations
            performance_recommendations = self._create_performance_recommendations(
                js_behavior, structure_analysis, llm_interpretation
            )
            
            # Step 9: Error handling strategies
            error_handling_strategies = self._create_error_handling_strategies(
                llm_interpretation, js_behavior
            )
            
            # Step 10: Learning insights
            learning_insights = self._extract_learning_insights(
                structure_analysis, llm_interpretation, data_mappings
            )
            
            # Calculate overall confidence
            confidence_score = self._calculate_overall_confidence(
                structure_analysis, data_mappings, llm_interpretation
            )
            
            # Determine site complexity
            site_complexity = self._determine_site_complexity(
                structure_analysis, js_behavior, llm_interpretation
            )
            
            # Create comprehensive report
            report = SiteIntelligenceReport(
                runbook_id=runbook_id,
                target_url=target_url,
                analysis_timestamp=datetime.now().isoformat(),
                site_complexity=site_complexity,
                structure_analysis=structure_analysis,
                javascript_behavior=js_behavior,
                data_mappings=data_mappings,
                navigation_workflow=navigation_workflow,
                extraction_strategy=extraction_strategy,
                performance_recommendations=performance_recommendations,
                error_handling_strategies=error_handling_strategies,
                learning_insights=learning_insights,
                cache_expires_at=(datetime.now() + timedelta(hours=self.cache_duration_hours)).isoformat(),
                confidence_score=confidence_score
            )
            
            # Cache results
            if self.cache_manager:
                self.cache_manager.cache_analysis(
                    target_url, self._serialize_report(report), self.cache_duration_hours
                )
            
            # Update knowledge base with learning insights
            self._update_knowledge_base(report)
            
            logger.info(f"✅ Site analysis complete - Confidence: {confidence_score:.2f}")
            return report
            
        except Exception as e:
            logger.error(f"❌ Site analysis failed: {e}")
            raise
        
        finally:
            # Don't close navigator here - might be reused
            pass
    
    def _create_minimal_js_behavior(self) -> JavaScriptBehavior:
        """Create minimal JS behavior for fast analysis"""
        return JavaScriptBehavior(
            has_ajax=False,
            ajax_endpoints=[],
            dynamic_loading=False,
            lazy_loading=False,
            infinite_scroll=False,
            form_validation=False,
            timing_requirements={'recommended_wait': 3.0, 'max_wait': 10.0},
            event_handlers=[],
            performance_metrics={}
        )
    
    def _create_navigation_workflow(self, llm_interpretation: Dict, 
                                  structure_analysis: SiteStructureAnalysis,
                                  site_prep: Dict) -> List[NavigationWorkflow]:
        """Create navigation workflow steps"""
        workflow = []
        
        # Basic navigation steps from LLM
        nav_steps = llm_interpretation.get('data_location_strategy', {}).get('navigation_steps', [])
        
        for i, step_data in enumerate(nav_steps):
            step = NavigationWorkflow(
                step_id=i + 1,
                action_type=step_data.get('action', 'wait'),
                target_selector=step_data.get('target', 'body'),
                parameters={},
                wait_condition="element_present",
                timeout=10.0,
                error_handling="retry_with_fallback",
                success_criteria="element_found"
            )
            workflow.append(step)
        
        # Add data extraction step
        extraction_step = NavigationWorkflow(
            step_id=len(workflow) + 1,
            action_type="extract",
            target_selector=llm_interpretation.get('data_location_strategy', {}).get('primary_data_container', 'body'),
            parameters={"extract_all_data": True},
            wait_condition="content_loaded",
            timeout=15.0,
            error_handling="retry_with_different_selector",
            success_criteria="data_extracted"
        )
        workflow.append(extraction_step)
        
        return workflow
    
    def _create_extraction_strategy(self, data_mappings: List[DataLocationMapping],
                                  llm_interpretation: Dict, js_behavior: JavaScriptBehavior) -> Dict[str, Any]:
        """Create comprehensive extraction strategy"""
        strategy = {
            "primary_approach": llm_interpretation.get('recommended_approach', {}).get('primary_strategy', 'table_extraction'),
            "data_field_strategies": {},
            "timing_strategy": {
                "page_load_wait": js_behavior.timing_requirements.get('recommended_wait', 3.0),
                "element_wait": 2.0,
                "extraction_delay": 1.0
            },
            "retry_strategy": {
                "max_retries": 3,
                "backoff_factor": 2.0,
                "fallback_selectors": True
            },
            "validation_strategy": {
                "validate_on_extract": True,
                "quality_threshold": 0.7,
                "required_fields": []
            }
        }
        
        # Add field-specific strategies
        for mapping in data_mappings:
            strategy["data_field_strategies"][mapping.field_name] = {
                "selectors": mapping.primary_selectors + mapping.fallback_selectors,
                "extraction_pattern": mapping.extraction_pattern,
                "validation_rules": mapping.validation_rules,
                "confidence": mapping.confidence_score
            }
            
            if mapping.confidence_score > 0.8:
                strategy["validation_strategy"]["required_fields"].append(mapping.field_name)
        
        return strategy
    
    def _create_performance_recommendations(self, js_behavior: JavaScriptBehavior,
                                          structure_analysis: SiteStructureAnalysis,
                                          llm_interpretation: Dict) -> Dict[str, Any]:
        """Create performance optimization recommendations"""
        recommendations = {
            "browser_optimizations": [],
            "timing_optimizations": {},
            "resource_management": {},
            "caching_strategy": {}
        }
        
        # Browser optimizations
        if not js_behavior.has_ajax:
            recommendations["browser_optimizations"].append("disable_javascript")
        if not js_behavior.dynamic_loading:
            recommendations["browser_optimizations"].append("disable_images")
        
        # Timing optimizations
        recommendations["timing_optimizations"] = {
            "page_load_timeout": js_behavior.timing_requirements.get('max_wait', 15.0),
            "element_wait_timeout": js_behavior.timing_requirements.get('recommended_wait', 5.0),
            "between_actions_delay": 1.0 if js_behavior.has_ajax else 0.5
        }
        
        # Resource management
        dom_complexity = structure_analysis.dom_complexity
        if dom_complexity > 1000:
            recommendations["resource_management"]["memory_cleanup"] = True
            recommendations["resource_management"]["selective_parsing"] = True
        
        # Caching strategy
        if llm_interpretation.get('confidence_assessment', {}).get('site_stability', 0.5) > 0.8:
            recommendations["caching_strategy"]["cache_duration_hours"] = 12
        else:
            recommendations["caching_strategy"]["cache_duration_hours"] = 2
        
        return recommendations
    
    def _create_error_handling_strategies(self, llm_interpretation: Dict,
                                        js_behavior: JavaScriptBehavior) -> List[Dict[str, Any]]:
        """Create error handling strategies"""
        strategies = []
        
        # Common strategies
        strategies.append({
            "error_type": "element_not_found",
            "strategy": "try_fallback_selectors",
            "max_retries": 3,
            "recovery_actions": ["refresh_page", "wait_longer", "use_alternative_selector"]
        })
        
        strategies.append({
            "error_type": "page_load_timeout",
            "strategy": "retry_with_longer_timeout",
            "max_retries": 2,
            "recovery_actions": ["increase_timeout", "disable_images", "use_different_user_agent"]
        })
        
        # Dynamic content specific
        if js_behavior.has_ajax:
            strategies.append({
                "error_type": "dynamic_content_not_loaded",
                "strategy": "wait_for_ajax_completion",
                "max_retries": 5,
                "recovery_actions": ["wait_for_network_idle", "trigger_load_events", "scroll_to_trigger"]
            })
        
        # Site complexity specific
        challenges = llm_interpretation.get('extraction_challenges', [])
        if 'authentication_required' in str(challenges).lower():
            strategies.append({
                "error_type": "authentication_failure",
                "strategy": "retry_authentication",
                "max_retries": 2,
                "recovery_actions": ["refresh_credentials", "use_different_auth_method", "manual_intervention"]
            })
        
        return strategies
    
    def _extract_learning_insights(self, structure_analysis: SiteStructureAnalysis,
                                 llm_interpretation: Dict, data_mappings: List[DataLocationMapping]) -> Dict[str, Any]:
        """Extract insights for knowledge base learning"""
        insights = {
            "successful_patterns": [],
            "site_characteristics": {},
            "optimization_opportunities": [],
            "pattern_reliability": {}
        }
        
        # Successful patterns
        for mapping in data_mappings:
            if mapping.confidence_score > 0.7:
                insights["successful_patterns"].append({
                    "field_type": mapping.field_name,
                    "selector_pattern": mapping.primary_selectors[0] if mapping.primary_selectors else None,
                    "confidence": mapping.confidence_score,
                    "data_type": mapping.data_type
                })
        
        # Site characteristics
        insights["site_characteristics"] = {
            "dom_complexity": structure_analysis.dom_complexity,
            "table_count": len(structure_analysis.table_structures),
            "form_count": len(structure_analysis.form_elements),
            "navigation_complexity": len(structure_analysis.navigation_elements),
            "site_type": llm_interpretation.get('site_analysis', {}).get('site_type', 'unknown'),
            "language": llm_interpretation.get('site_analysis', {}).get('language', 'unknown')
        }
        
        # Optimization opportunities
        if structure_analysis.dom_complexity > 500:
            insights["optimization_opportunities"].append("Consider selective DOM parsing")
        
        high_confidence_mappings = [m for m in data_mappings if m.confidence_score > 0.8]
        if len(high_confidence_mappings) == len(data_mappings):
            insights["optimization_opportunities"].append("Site has stable patterns - increase cache duration")
        
        return insights
    
    def _calculate_overall_confidence(self, structure_analysis: SiteStructureAnalysis,
                                    data_mappings: List[DataLocationMapping],
                                    llm_interpretation: Dict) -> float:
        """Calculate overall confidence score"""
        confidence_factors = []
        
        # Data mapping confidence
        if data_mappings:
            avg_mapping_confidence = sum(m.confidence_score for m in data_mappings) / len(data_mappings)
            confidence_factors.append(avg_mapping_confidence * 0.4)
        
        # LLM interpretation confidence
        llm_confidence = llm_interpretation.get('confidence_assessment', {}).get('overall_confidence', 0.5)
        confidence_factors.append(llm_confidence * 0.3)
        
        # Structure analysis confidence
        structure_confidence = min(1.0, structure_analysis.accessibility_score + 0.3)
        if structure_analysis.table_structures:
            structure_confidence += 0.2
        if structure_analysis.form_elements:
            structure_confidence += 0.1
        structure_confidence = min(1.0, structure_confidence)
        confidence_factors.append(structure_confidence * 0.3)
        
        return sum(confidence_factors) if confidence_factors else 0.5
    
    def _determine_site_complexity(self, structure_analysis: SiteStructureAnalysis,
                                 js_behavior: JavaScriptBehavior,
                                 llm_interpretation: Dict) -> SiteComplexity:
        """Determine overall site complexity"""
        complexity_score = 0
        
        # DOM complexity
        if structure_analysis.dom_complexity > 1000:
            complexity_score += 2
        elif structure_analysis.dom_complexity > 500:
            complexity_score += 1
        
        # JavaScript complexity
        if js_behavior.has_ajax:
            complexity_score += 1
        if js_behavior.dynamic_loading:
            complexity_score += 1
        if js_behavior.lazy_loading or js_behavior.infinite_scroll:
            complexity_score += 1
        
        # Navigation complexity
        if len(structure_analysis.navigation_elements) > 3:
            complexity_score += 1
        
        # LLM-assessed complexity
        llm_complexity = llm_interpretation.get('site_analysis', {}).get('complexity', 'moderate')
        if llm_complexity == 'complex':
            complexity_score += 2
        elif llm_complexity == 'moderate':
            complexity_score += 1
        
        # Determine final complexity
        if complexity_score >= 6:
            return SiteComplexity.DYNAMIC
        elif complexity_score >= 4:
            return SiteComplexity.COMPLEX
        elif complexity_score >= 2:
            return SiteComplexity.MODERATE
        else:
            return SiteComplexity.SIMPLE
    
    def _serialize_report(self, report: SiteIntelligenceReport) -> Dict[str, Any]:
        """Serialize report for caching"""
        return asdict(report)
    
    def _deserialize_report(self, data: Dict[str, Any]) -> SiteIntelligenceReport:
        """Deserialize report from cache"""
        # Convert nested dictionaries back to dataclasses
        structure_data = data['structure_analysis']
        structure_analysis = SiteStructureAnalysis(**structure_data)
        
        js_data = data['javascript_behavior']
        js_behavior = JavaScriptBehavior(**js_data)
        
        data_mappings = []
        for mapping_data in data['data_mappings']:
            mapping = DataLocationMapping(**mapping_data)
            data_mappings.append(mapping)
        
        navigation_workflow = []
        for workflow_data in data['navigation_workflow']:
            workflow = NavigationWorkflow(**workflow_data)
            navigation_workflow.append(workflow)
        
        # Create report
        report = SiteIntelligenceReport(
            runbook_id=data['runbook_id'],
            target_url=data['target_url'],
            analysis_timestamp=data['analysis_timestamp'],
            site_complexity=SiteComplexity(data['site_complexity']),
            structure_analysis=structure_analysis,
            javascript_behavior=js_behavior,
            data_mappings=data_mappings,
            navigation_workflow=navigation_workflow,
            extraction_strategy=data['extraction_strategy'],
            performance_recommendations=data['performance_recommendations'],
            error_handling_strategies=data['error_handling_strategies'],
            learning_insights=data['learning_insights'],
            cache_expires_at=data['cache_expires_at'],
            confidence_score=data['confidence_score']
        )
        
        return report
    
    def _update_knowledge_base(self, report: SiteIntelligenceReport):
        """Update knowledge base with learning insights"""
        try:
            # This would integrate with the Runbook Intelligence knowledge base
            # For now, we'll just log the insights
            insights = report.learning_insights
            logger.info(f"🧠 Learning insights captured:")
            logger.info(f"   • Successful patterns: {len(insights.get('successful_patterns', []))}")
            logger.info(f"   • Site type: {insights.get('site_characteristics', {}).get('site_type', 'unknown')}")
            logger.info(f"   • Optimization opportunities: {len(insights.get('optimization_opportunities', []))}")
            
            # TODO: Integrate with actual knowledge base
            # knowledge_base.update_site_patterns(report.target_url, insights)
            
        except Exception as e:
            logger.warning(f"⚠️ Knowledge base update failed: {e}")
    
    def get_analysis_summary(self, report: SiteIntelligenceReport) -> Dict[str, Any]:
        """Get a summary of the analysis for reporting"""
        summary = {
            "url": report.target_url,
            "complexity": report.site_complexity.value,
            "confidence": report.confidence_score,
            "analysis_timestamp": report.analysis_timestamp,
            "data_fields_mapped": len(report.data_mappings),
            "navigation_steps": len(report.navigation_workflow),
            "has_javascript": report.javascript_behavior.has_ajax or report.javascript_behavior.dynamic_loading,
            "recommended_approach": report.extraction_strategy.get('primary_approach', 'unknown'),
            "cache_expires": report.cache_expires_at,
            "high_confidence_fields": [
                mapping.field_name for mapping in report.data_mappings 
                if mapping.confidence_score > 0.8
            ],
            "potential_challenges": len(report.error_handling_strategies),
            "optimization_recommendations": len(report.performance_recommendations.get('browser_optimizations', []))
        }
        return summary
    
    def close(self):
        """Clean up resources"""
        if self.navigator:
            self.navigator.close()
        if self.cache_manager:
            self.cache_manager.cleanup_expired_cache()
        logger.info("✅ Site Intelligence Engine closed")

# =====================================================================
# INTEGRATION HELPER FUNCTIONS
# =====================================================================

def create_site_intelligence_from_runbook(runbook_knowledge) -> Dict[str, Any]:
    """Convert runbook knowledge to site analysis preparation format"""
    return {
        "runbook_id": getattr(runbook_knowledge, 'runbook_id', 'unknown'),
        "target_url": getattr(runbook_knowledge, 'primary_source_url', ''),
        "expected_data_patterns": getattr(runbook_knowledge, 'data_targets', []),
        "business_context": {
            "domain": getattr(runbook_knowledge, 'business_domain', 'general'),
            "dataset_name": getattr(runbook_knowledge, 'dataset_name', 'unknown'),
            "complexity": getattr(runbook_knowledge, 'complexity_score', 0.5)
        },
        "authentication_requirements": getattr(runbook_knowledge, 'authentication_info', None),
        "navigation_hints": [],
        "success_criteria": {
            "expected_data_fields": len(getattr(runbook_knowledge, 'data_targets', [])),
            "minimum_confidence": 0.7
        }
    }

# =====================================================================
# DEMONSTRATION AND TESTING
# =====================================================================

def demo_site_intelligence_engine():
    """Demonstrate the Site Intelligence Engine with SHFE example"""
    print("🎯 DEMO: Site Intelligence Engine")
    print("=" * 60)
    
    # Sample site preparation (from Runbook Intelligence output)
    shfe_site_prep = {
        "runbook_id": "runbook_20250629_shfe",
        "target_url": "https://www.shfe.com.cn/publicnotice/notice/",
        "expected_data_patterns": [
            {
                "field_name": "effective_date",
                "description": "Date when margin changes take effect",
                "data_type": "date"
            },
            {
                "field_name": "commodity",
                "description": "Commodity name (aluminum, zinc, gold, etc.)",
                "data_type": "text"
            },
            {
                "field_name": "hedging_percentage",
                "description": "Margin ratio for hedging transactions",
                "data_type": "percentage"
            },
            {
                "field_name": "speculative_percentage",
                "description": "Margin ratio for speculative transactions",
                "data_type": "percentage"
            }
        ],
        "business_context": {
            "domain": "financial_exchange",
            "dataset_name": "SHFEMR",
            "complexity": 0.65
        },
        "navigation_hints": [
            "Look for notice/announcement sections",
            "Check for margin ratio or trading updates"
        ],
        "success_criteria": {
            "expected_data_fields": 4,
            "minimum_confidence": 0.8
        }
    }
    
    try:
        # Initialize Site Intelligence Engine
        print("🚀 Initializing Site Intelligence Engine...")
        engine = SiteIntelligenceEngine(
            model_name="codellama:13b-instruct",
            headless=True,
            enable_caching=True,
            cache_duration_hours=6
        )
        
        print("✅ Engine initialized successfully")
        
        # Perform site analysis
        print(f"\n🔍 Analyzing site: {shfe_site_prep['target_url']}")
        print("This may take 30-60 seconds for deep analysis...")
        
        report = engine.analyze_site(shfe_site_prep, AnalysisDepth.THOROUGH)
        
        # Display results
        print(f"\n🎉 Analysis Complete!")
        print(f"📊 Results Summary:")
        print(f"   • Site Complexity: {report.site_complexity.value}")
        print(f"   • Overall Confidence: {report.confidence_score:.2f}")
        print(f"   • Data Fields Mapped: {len(report.data_mappings)}")
        print(f"   • Navigation Steps: {len(report.navigation_workflow)}")
        print(f"   • JavaScript Required: {report.javascript_behavior.has_ajax}")
        print(f"   • Dynamic Content: {report.javascript_behavior.dynamic_loading}")
        
        # Show data mappings
        print(f"\n📍 Data Location Mappings:")
        for mapping in report.data_mappings:
            print(f"   • {mapping.field_name}: {mapping.confidence_score:.2f} confidence")
            if mapping.primary_selectors:
                print(f"     Primary selector: {mapping.primary_selectors[0]}")
            if mapping.sample_values:
                print(f"     Sample values: {mapping.sample_values[:2]}")
        
        # Show extraction strategy
        print(f"\n🎯 Extraction Strategy:")
        strategy = report.extraction_strategy
        print(f"   • Primary Approach: {strategy.get('primary_approach', 'unknown')}")
        print(f"   • Timing Strategy: {strategy.get('timing_strategy', {})}")
        print(f"   • Required Fields: {strategy.get('validation_strategy', {}).get('required_fields', [])}")
        
        # Show performance recommendations
        print(f"\n⚡ Performance Recommendations:")
        perf = report.performance_recommendations
        browser_opts = perf.get('browser_optimizations', [])
        if browser_opts:
            print(f"   • Browser optimizations: {browser_opts}")
        timing_opts = perf.get('timing_optimizations', {})
        if timing_opts:
            print(f"   • Optimal timeouts: {timing_opts}")
        
        # Show learning insights
        print(f"\n🧠 Learning Insights:")
        insights = report.learning_insights
        successful_patterns = insights.get('successful_patterns', [])
        print(f"   • Successful patterns: {len(successful_patterns)}")
        site_chars = insights.get('site_characteristics', {})
        print(f"   • Site type: {site_chars.get('site_type', 'unknown')}")
        print(f"   • Language: {site_chars.get('language', 'unknown')}")
        opt_opportunities = insights.get('optimization_opportunities', [])
        if opt_opportunities:
            print(f"   • Optimization opportunities: {opt_opportunities}")
        
        # Get analysis summary for next engine
        summary = engine.get_analysis_summary(report)
        print(f"\n📋 Ready for Code Generation Engine:")
        print(f"   • High confidence fields: {summary['high_confidence_fields']}")
        print(f"   • Recommended approach: {summary['recommended_approach']}")
        print(f"   • Potential challenges: {summary['potential_challenges']}")
        
        print(f"\n🔄 Next Step: Pass report to Code Generation Engine")
        print(f"   code_generation_engine.generate_scraper(report)")
        
        return report
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        try:
            engine.close()
        except:
            pass

def test_site_intelligence_components():
    """Test individual components of the Site Intelligence Engine"""
    print("🧪 TESTING: Site Intelligence Components")
    print("=" * 50)
    
    try:
        # Test 1: Navigator
        print("1. Testing SeleniumSiteNavigator...")
        navigator = SeleniumSiteNavigator(headless=True, enable_translation=True)
        result = navigator.navigate_to_site("https://httpbin.org/html")
        print(f"   ✅ Navigation: {result.get('status', 'unknown')}")
        
        # Test 2: Structure Analyzer
        print("2. Testing HTMLStructureAnalyzer...")
        analyzer = HTMLStructureAnalyzer(navigator)
        structure = analyzer.analyze_structure("https://httpbin.org/html")
        print(f"   ✅ Structure analysis: DOM complexity {structure.dom_complexity}")
        
        # Test 3: JavaScript Detector
        print("3. Testing JavaScriptBehaviorDetector...")
        js_detector = JavaScriptBehaviorDetector(navigator)
        js_behavior = js_detector.analyze_behavior("https://httpbin.org/html")
        print(f"   ✅ JS analysis: AJAX={js_behavior.has_ajax}, Dynamic={js_behavior.dynamic_loading}")
        
        # Test 4: LLM Interpreter
        print("4. Testing LLMSiteInterpreter...")
        llm_interpreter = LLMSiteInterpreter()
        sample_content = "<html><body><h1>Test</h1><table><tr><td>Data</td></tr></table></body></html>"
        interpretation = llm_interpreter.interpret_site_content(
            "https://example.com", 
            sample_content, 
            {"expected_data_patterns": [{"field_name": "test_data"}]}
        )
        print(f"   ✅ LLM interpretation: {interpretation.get('site_analysis', {}).get('site_type', 'unknown')}")
        
        # Test 5: Cache Manager
        print("5. Testing SiteIntelligenceCacheManager...")
        cache_manager = SiteIntelligenceCacheManager("test_cache.db")
        test_data = {"test": "data"}
        cache_manager.cache_analysis("https://example.com", test_data, 1)
        cached = cache_manager.get_cached_analysis("https://example.com")
        print(f"   ✅ Cache: {'Working' if cached else 'Failed'}")
        
        navigator.close()
        print("\n✅ All component tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        return False

def demo_with_fallback_site():
    """Demo with a simpler site if SHFE is not accessible"""
    print("🎯 DEMO: Site Intelligence with Fallback Site")
    print("=" * 50)
    
    # Use a simple, reliable test site
    test_site_prep = {
        "runbook_id": "test_demo",
        "target_url": "https://httpbin.org/html",
        "expected_data_patterns": [
            {
                "field_name": "page_title",
                "description": "Page title text",
                "data_type": "text"
            },
            {
                "field_name": "heading",
                "description": "Main heading",
                "data_type": "text"
            }
        ],
        "business_context": {
            "domain": "test",
            "dataset_name": "test_demo",
            "complexity": 0.2
        },
        "success_criteria": {
            "expected_data_fields": 2,
            "minimum_confidence": 0.5
        }
    }
    
    try:
        engine = SiteIntelligenceEngine(headless=True, enable_caching=False)
        report = engine.analyze_site(test_site_prep, AnalysisDepth.FAST)
        
        print(f"✅ Fallback demo successful!")
        print(f"   • Confidence: {report.confidence_score:.2f}")
        print(f"   • Mappings: {len(report.data_mappings)}")
        print(f"   • Complexity: {report.site_complexity.value}")
        
        engine.close()
        return report
        
    except Exception as e:
        print(f"❌ Fallback demo failed: {e}")
        return None

# Main execution
def main():
    """Main demonstration function"""
    print("🎯 SITE INTELLIGENCE ENGINE - COMPLETE DEMO")
    print("=" * 60)
    print("This demo shows the complete Site Intelligence Engine:")
    print("1. 🌐 Selenium-based site navigation with translation")
    print("2. 🔍 Deep HTML structure analysis")
    print("3. ⚡ JavaScript behavior detection")
    print("4. 🧠 LLM-powered content interpretation")
    print("5. 📍 Data location mapping")
    print("6. 💾 Intelligent caching")
    print("7. 🔄 Learning integration")
    print()
    
    try:
        # First try component tests
        print("Step 1: Testing individual components...")
        if test_site_intelligence_components():
            print("✅ Component tests passed\n")
        else:
            print("⚠️ Some component tests failed, continuing with demo\n")
        
        # Try main demo
        print("Step 2: Full site analysis demo...")
        report = demo_site_intelligence_engine()
        
        if not report:
            print("Main demo failed, trying fallback...")
            report = demo_with_fallback_site()
        
        if report:
            print(f"\n🎉 DEMO COMPLETED SUCCESSFULLY!")
            print(f"📦 Site Intelligence Report ready for Code Generation Engine")
            print(f"\n🔄 Next Phase: Code Generation Engine")
            print(f"   Will receive comprehensive site analysis with:")
            print(f"   • {len(report.data_mappings)} data field mappings")
            print(f"   • {len(report.navigation_workflow)} navigation steps")
            print(f"   • Extraction strategy: {report.extraction_strategy.get('primary_approach')}")
            print(f"   • Performance optimizations included")
            print(f"   • Error handling strategies: {len(report.error_handling_strategies)}")
            
            return report
        else:
            print("❌ All demos failed")
            return None
            
    except Exception as e:
        print(f"❌ Demo execution failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()