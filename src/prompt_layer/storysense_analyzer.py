import json
import configparser
import os
import logging
from src.llm_layer.model_manual_test_llm import LLM
# from src.configuration_handler.env_manager import EnvManager
from src.metrics.metrics_manager import MetricsManager

# Set up logging
logging.basicConfig(level=logging.INFO)


class StorySenseAnalyzer:
    def __init__(self, llm_family, metrics_manager=None):
        # Initialize environment manager
        # self.env_manager = EnvManager()

        # Use provided metrics_manager or create a new one
        print(f"DEBUG: StorySenseAnalyzer received metrics_manager: {id(metrics_manager)}")
        self.metrics_manager = metrics_manager or MetricsManager

        # Use environment variable for LLM family if not provided
        if llm_family is None:
            llm_family = os.getenv('LLM_FAMILY', 'AWS')

        # Pass metrics_manager to LLM
        self.llm = LLM(llm_family, metrics_manager=self.metrics_manager)
        self.llm_family = llm_family
        self.config_path = '../Config'

        # Load AWS config if using AWS
        if llm_family == 'AWS':
            self.config_parser_aws = configparser.ConfigParser()
            self.config_parser_aws.read(os.path.join(self.config_path, 'ConfigAWS.properties'))

    def analyze_user_story(self, user_story_data):
        """
        Analyze a user story using LLM and return insights
        """
        us_id = user_story_data["us_id"]
        userstory = user_story_data["userstory"]
        context = user_story_data["context"]
        context_file_types = user_story_data.get("context_file_types", {})

        # Updated prompt template with reference example, enhanced output structure, and file type info
        prompt_template = """
        You are an expert user story analyzer. Your task is to analyze the following user story and provide insights.

        USER STORY:
        {userstory}

        {context_prompt}

        Use this high-quality user story as a reference for scoring (rated 10 for each quality parameter):
        Title: Online payment option for utility bills 
        Description: As a registered user, I want to be able to pay all my utility bills (such as water, electricity, and gas) online so that I can avoid the hassle of manual payment methods and keep track of my previous payments. 
        Acceptance Criteria: 
        1. The user should be able to view all utility bills the moment they log in. 
        2. The user should be able to select any of utility bills listed and see the detailed bill. 
        3. The system should provide the option for online payment for the selected bill. 
        4. Once the payment is successful, the system should update the bill status and provide the user with a receipt. 
        5. The user should be allowed to download and save the receipt for their records. 
        6. The user should be able to view his payment history.

        Please analyze this user story and evaluate it on the following 10 characteristics:

        1. User-Centered: Does the story focus on the user's needs and perspective?
        2. Independent: Can this story be developed independently of other stories?
        3. Negotiable: Is there room for discussion about implementation details?
        4. Valuable: Does the story deliver clear value to users or stakeholders?
        5. Estimable: Can developers reasonably estimate the effort required?
        6. Small: Is the story appropriately sized for a single iteration?
        7. Testable: Are there clear criteria to verify when it's done?
        8. Acceptance Criteria: Are the acceptance criteria clear and comprehensive?
        9. Prioritized: Is the importance of this story clear relative to others?
        10. Collaboration and Understanding: Does the story facilitate team understanding?

        For each characteristic, provide:
        - A score from 1-10 (where 10 is highest quality)
        - A justification for the score (only if score is less than 10, otherwise say "N/A")
        - Recommendations for improvement (only if score is less than 10, otherwise say "N/A")
          Include specific examples of how the user story could be rephrased or modified

        Also, based on the context and your analysis, create a reframed version of the user story that addresses all the identified issues.

        Format your response as JSON with the following structure:
        {{
            "recommendation": {{
                "recommended": "yes if there is a change in the newUserstory from the original else no",
                "descriptionDiff": "difference in newUserstory and original userstory",
                "acceptanceCriteriaDiff": "the difference in the acceptance criteria of newUserStory and original user story",
                "newUserStory": "newUserstory with acceptance criteria using recommendations and context"
            }},
            "user_centered_score": <score>,
            "user_centered_justification": "<explanation>",
            "user_centered_recommendations": ["<recommendation1>", "<recommendation2>", ...],

            "independent_score": <score>,
            "independent_justification": "<explanation>",
            "independent_recommendations": ["<recommendation1>", "<recommendation2>", ...],

            "negotiable_score": <score>,
            "negotiable_justification": "<explanation>",
            "negotiable_recommendations": ["<recommendation1>", "<recommendation2>", ...],

            "valuable_score": <score>,
            "valuable_justification": "<explanation>",
            "valuable_recommendations": ["<recommendation1>", "<recommendation2>", ...],

            "estimable_score": <score>,
            "estimable_justification": "<explanation>",
            "estimable_recommendations": ["<recommendation1>", "<recommendation2>", ...],

            "small_score": <score>,
            "small_justification": "<explanation>",
            "small_recommendations": ["<recommendation1>", "<recommendation2>", ...],

            "testable_score": <score>,
            "testable_justification": "<explanation>",
            "testable_recommendations": ["<recommendation1>", "<recommendation2>", ...],

            "acceptance_criteria_score": <score>,
            "acceptance_criteria_justification": "<explanation>",
            "acceptance_criteria_recommendations": ["<recommendation1>", "<recommendation2>", ...],

            "prioritized_score": <score>,
            "prioritized_justification": "<explanation>",
            "prioritized_recommendations": ["<recommendation1>", "<recommendation2>", ...],

            "collaboration_score": <score>,
            "collaboration_justification": "<explanation>",
            "collaboration_recommendations": ["<recommendation1>", "<recommendation2>", ...],

            "key_insights": ["<insight1>", "<insight2>", ...],
            "potential_risks": ["<risk1>", "<risk2>", ...],
            "recommendations": ["<recommendation1>", "<recommendation2>", ...],
            "context_analysis": {{
                "most_useful_file_types": ["<file_type1>", "<file_type2>"],
                "context_quality_assessment": "<assessment of how useful the context was>"
            }}
        }}
        """

        # Update how you format the context_prompt:
        if context:
            # Log context usage for visibility
            from colorama import Fore, Style
            context_length = len(str(context))
            print(f"{Fore.CYAN}🔍 Using context for story analysis:{Style.RESET_ALL}")
            print(f"   📊 Context length: {context_length:,} characters")
            
            # Format file type information
            file_type_summary = ""
            if context_file_types:
                file_type_summary = "\nContext sources include:\n"
                context_sources = []
                for file_type, count in context_file_types.items():
                    file_type_summary += f"- {count} {file_type} document(s)\n"
                    context_sources.append(f"{count} {file_type}")
                print(f"   📁 Sources: {', '.join(context_sources)}")
            
            # Show a preview of the context content
            context_preview = str(context)[:300] + "..." if len(str(context)) > 300 else str(context)
            print(f"   📝 Context preview: {context_preview}")

            context_prompt = f"""
            ADDITIONAL CONTEXT:
            {context}
            {file_type_summary}

            Use the above context to enhance your analysis. Pay special attention to:
            - Business rules and constraints mentioned in the context
            - Similar requirements or user stories
            - Relevant terminology and definitions
            - Company policies or standards that apply
            - Consider the reliability of different document types in your analysis
            """
        else:
            from colorama import Fore, Style
            print(f"{Fore.YELLOW}⚠️  No additional context available for this user story{Style.RESET_ALL}")
            context_prompt = "No additional context is available for this user story."

        # For AWS/Claude, add explicit JSON formatting instructions
        if self.llm_family == 'AWS':
            prompt_template += """
            CRITICAL FORMATTING REQUIREMENTS:
            1. Your response MUST be valid JSON only - no explanatory text, no markdown formatting, no code blocks
            2. Start your response immediately with the opening curly brace {
            3. End your response with the closing curly brace }
            4. Ensure all JSON keys and string values are properly quoted with double quotes
            5. Use only double quotes, never single quotes
            6. Do not include any text before or after the JSON object
            7. If you need to include quotes within string values, escape them with backslashes
            
            Example of correct format:
            {"key": "value", "score": 5, "list": ["item1", "item2"]}
            
            Example of INCORRECT format:
            Here's the analysis: {"key": "value"}
            ```json
            {"key": "value"}
            ```
            """

        # Send request to LLM
        input_variables = ["userstory", "context_prompt"]
        input_variables_dict = {
            "userstory": userstory,
            "context_prompt": context_prompt
        }

        logging.info(f"Sending user story {us_id} to LLM for analysis with file type info")

        # Track start time for performance metrics
        import time
        start_time = time.time()

        # Call LLM with call_type for metrics tracking
        response = self.llm.send_request_multimodal(prompt_template,image_path="", call_type="analysis")
        
        # Enhanced logging for debugging
        logging.info(f"Received response from LLM: {len(response)} characters")
        logging.info(f"Response preview (first 500 chars): {response[:500]}")
        logging.info(f"Response ending (last 200 chars): {response[-200:]}")
        logging.info(f"Contains opening brace: {'{' in response}")
        logging.info(f"Contains closing brace: {'}' in response}")
        logging.info(f"Contains 'json': {'json' in response.lower()}")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        logging.info(f"Received response from LLM for user story {us_id} in {processing_time:.2f}s")
        
        # Also save the raw response to a file for debugging
        debug_file = f"../Output/debug_response_{us_id}_{int(time.time())}.txt"
        try:
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write(f"User Story ID: {us_id}\n")
                f.write(f"Processing Time: {processing_time:.2f}s\n")
                f.write(f"Response Length: {len(response)}\n")
                f.write("-" * 50 + "\n")
                f.write(response)
            logging.info(f"Saved raw response to {debug_file}")
        except Exception as e:
            logging.warning(f"Could not save debug file: {e}")

        # Parse JSON response with improved error handling
        try:
            analysis = self._extract_and_parse_json(response)
            
            # If extraction failed, try a simplified prompt
            if (isinstance(analysis, dict) and 
                analysis.get('user_centered_justification', '').startswith('Could not analyze')):
                
                logging.warning("Complex prompt failed, trying simplified approach")
                analysis = self._try_simplified_analysis(user_story_data)
                
        except Exception as e:
            logging.error(f"Error parsing LLM response: {str(e)}")
            analysis = self._create_fallback_analysis(f"Error parsing response: {str(e)}")

        # Add raw LLM response for debugging (truncated to avoid huge files)
        analysis["raw_llm_response"] = response[:1000] + ("..." if len(response) > 1000 else "")

        # Add user story information
        analysis["us_id"] = us_id
        analysis["userstory"] = userstory

        # Add context information
        analysis["context_quality"] = user_story_data.get("context_quality", "none")
        analysis["context_count"] = user_story_data.get("context_count", 0)
        analysis["context_file_types"] = context_file_types  # Add file type information

        # Add performance metrics
        analysis["processing_time"] = processing_time

        # Ensure all required fields are present
        self._ensure_required_fields(analysis)

        # Calculate overall score - now average of all 10 parameters
        analysis["overall_score"] = (
                                            analysis["user_centered_score"] +
                                            analysis["independent_score"] +
                                            analysis["negotiable_score"] +
                                            analysis["valuable_score"] +
                                            analysis["estimable_score"] +
                                            analysis["small_score"] +
                                            analysis["testable_score"] +
                                            analysis["acceptance_criteria_score"] +
                                            analysis["prioritized_score"] +
                                            analysis["collaboration_score"]
                                    ) / 10

        logging.info(f"Analysis completed for user story {us_id} with overall score {analysis['overall_score']}")
        return analysis

    def _extract_and_parse_json(self, response):
        """
        Extract and parse JSON from LLM response with multiple fallback strategies
        """
        # Strategy 1: Look for JSON blocks marked with ```json
        if '```json' in response:
            try:
                json_content = response.split('```json')[1].split('```')[0].strip()
                logging.info("Found JSON content in markdown code block")
                return json.loads(json_content)
            except (json.JSONDecodeError, IndexError) as e:
                logging.warning(f"Failed to parse JSON from markdown block: {e}")

        # Strategy 2: Look for JSON blocks marked with ``` 
        if '```' in response and '{' in response:
            try:
                # Find content between triple backticks that contains JSON
                blocks = response.split('```')
                for block in blocks:
                    block = block.strip()
                    if block.startswith('{') and block.endswith('}'):
                        logging.info("Found JSON content in generic code block")
                        return json.loads(block)
            except json.JSONDecodeError as e:
                logging.warning(f"Failed to parse JSON from code block: {e}")

        # Strategy 3: Find the largest JSON object in the response
        json_start = response.find('{')
        if json_start >= 0:
            # Find the matching closing brace
            brace_count = 0
            json_end = json_start
            for i, char in enumerate(response[json_start:], json_start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break
            
            if json_end > json_start:
                try:
                    json_content = response[json_start:json_end]
                    logging.info("Found JSON content using brace matching")
                    return json.loads(json_content)
                except json.JSONDecodeError as e:
                    logging.warning(f"Failed to parse JSON using brace matching: {e}")

        # Strategy 4: Clean up response and try parsing the whole thing
        try:
            # Remove any leading/trailing non-JSON content
            cleaned_response = response.strip()
            
            # Remove common prefixes that Claude might add
            prefixes_to_remove = [
                "Here's the analysis:",
                "Here is the analysis:",
                "Analysis:",
                "```json",
                "```"
            ]
            
            for prefix in prefixes_to_remove:
                if cleaned_response.startswith(prefix):
                    cleaned_response = cleaned_response[len(prefix):].strip()
            
            # Remove common suffixes
            suffixes_to_remove = ["```"]
            for suffix in suffixes_to_remove:
                if cleaned_response.endswith(suffix):
                    cleaned_response = cleaned_response[:-len(suffix)].strip()
            
            # Try to parse the cleaned response
            if cleaned_response.startswith('{') and cleaned_response.endswith('}'):
                logging.info("Attempting to parse cleaned response as JSON")
                return json.loads(cleaned_response)
        
        except json.JSONDecodeError as e:
            logging.warning(f"Failed to parse cleaned response: {e}")

        # Strategy 5: Extract JSON-like content using regex
        import re
        try:
            # Look for patterns that look like JSON objects
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.findall(json_pattern, response, re.DOTALL)
            
            for match in matches:
                try:
                    parsed = json.loads(match)
                    # Check if this looks like our expected structure
                    if isinstance(parsed, dict) and 'user_centered_score' in parsed:
                        logging.info("Found valid JSON using regex pattern matching")
                        return parsed
                except json.JSONDecodeError:
                    continue
        
        except Exception as e:
            logging.warning(f"Regex extraction failed: {e}")

        # Strategy 6: If response indicates an error, try to regenerate with simpler prompt
        if any(error_indicator in response.lower() for error_indicator in ['error', 'unable', 'cannot', 'failed', 'sorry']):
            logging.warning("Response contains error indicators, attempting simplified analysis")
            return self._generate_simple_analysis(response)
        
        # Strategy 7: Last resort - create a JSON-like structure from any scoring information found
        if any(word in response.lower() for word in ['score', 'rating', 'analysis', 'recommendation']):
            logging.warning("Attempting to extract partial analysis from response")
            return self._extract_partial_analysis(response)
        
        # If all strategies fail, log the response and create fallback
        logging.error("All JSON extraction strategies failed")
        logging.error(f"Raw response (first 500 chars): {response[:500]}")
        logging.error(f"Raw response (last 200 chars): {response[-200:]}")
        logging.error(f"Response contains curly braces: {'{' in response and '}' in response}")
        logging.error(f"Response contains 'json': {'json' in response.lower()}")
        logging.error(f"Response type: {type(response)}")
        logging.error(f"Response is empty: {len(response.strip()) == 0}")
        
        return self._create_fallback_analysis(f"Unable to parse LLM response - all JSON extraction strategies failed. Response length: {len(response)}")

    def _generate_simple_analysis(self, error_response):
        """Generate a simple analysis when the LLM returns an error"""
        logging.info("Generating simple analysis due to LLM error response")
        
        reason = "LLM returned an error response"
        if "error" in error_response.lower():
            reason = "LLM error: " + error_response[:200]
        
        return self._create_fallback_analysis(reason)

    def _extract_partial_analysis(self, response):
        """Try to extract any scoring or analysis information from a non-JSON response"""
        import re
        
        logging.info("Attempting to extract partial analysis from non-JSON response")
        
        # Try to find numerical scores in the response
        score_patterns = [
            r'(\w+).*?score.*?(\d+)',
            r'score.*?(\w+).*?(\d+)',
            r'(\w+).*?(\d+)\/10',
            r'(\d+).*?(\w+)',
        ]
        
        extracted_scores = {}
        score_fields = [
            'user_centered', 'independent', 'negotiable', 'valuable', 
            'estimable', 'small', 'testable', 'acceptance_criteria', 
            'prioritized', 'collaboration'
        ]
        
        for pattern in score_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for match in matches:
                if len(match) == 2:
                    word, score = match
                    try:
                        score_val = int(score)
                        if 1 <= score_val <= 10:
                            for field in score_fields:
                                if field.lower() in word.lower() or word.lower() in field.lower():
                                    extracted_scores[f"{field}_score"] = score_val
                                    break
                    except ValueError:
                        continue
        
        if extracted_scores:
            logging.info(f"Extracted scores: {extracted_scores}")
            analysis = self._create_fallback_analysis("Partial analysis extracted from non-JSON response")
            analysis.update(extracted_scores)
            return analysis
        
        return self._create_fallback_analysis("Could not extract any scoring information from response")

    def _try_simplified_analysis(self, user_story_data):
        """Try a much simpler prompt if the complex one fails"""
        
        us_id = user_story_data["us_id"]
        userstory = user_story_data["userstory"]
        
        simple_prompt = f"""
        Analyze this user story and respond with ONLY valid JSON. No other text.

        User Story: {userstory}

        Respond with exactly this JSON structure:
        {{
            "user_centered_score": 5,
            "independent_score": 5,
            "negotiable_score": 5,
            "valuable_score": 5,
            "estimable_score": 5,
            "small_score": 5,
            "testable_score": 5,
            "acceptance_criteria_score": 5,
            "prioritized_score": 5,
            "collaboration_score": 5,
            "recommendation": {{
                "recommended": "no",
                "descriptionDiff": "No changes needed",
                "acceptanceCriteriaDiff": "No changes needed", 
                "newUserStory": "{userstory}"
            }}
        }}
        
        Replace the scores (1-10) based on your analysis. Start response with {{ and end with }}.
        """
        
        try:
            logging.info(f"Trying simplified analysis for user story {us_id}")
            response = self.llm.send_request_multimodal(simple_prompt, image_path="", call_type="simplified_analysis")
            
            logging.info(f"Simplified response length: {len(response)}")
            logging.info(f"Simplified response preview: {response[:200]}")
            
            # Try to parse the simplified response
            analysis = self._extract_and_parse_json(response)
            
            if isinstance(analysis, dict) and 'user_centered_score' in analysis:
                logging.info("Simplified analysis succeeded!")
                
                # Add the missing fields
                analysis["us_id"] = us_id
                analysis["userstory"] = userstory
                analysis["context_quality"] = user_story_data.get("context_quality", "none")
                analysis["context_count"] = user_story_data.get("context_count", 0)
                analysis["context_file_types"] = user_story_data.get("context_file_types", {})
                analysis["processing_time"] = 0
                
                # Ensure all required fields
                self._ensure_required_fields(analysis)
                
                return analysis
            else:
                logging.warning("Simplified analysis also failed")
                return self._create_fallback_analysis("Both complex and simplified analysis failed")
                
        except Exception as e:
            logging.error(f"Simplified analysis error: {e}")
            return self._create_fallback_analysis(f"Simplified analysis error: {str(e)}")

    def _create_fallback_analysis(self, reason):
        """Create a fallback analysis when parsing fails"""
        logging.warning(f"Creating fallback analysis due to: {reason}")
        return {
            "recommendation": {
                "recommended": "no",
                "descriptionDiff": f"Could not analyze: {reason}",
                "acceptanceCriteriaDiff": f"Could not analyze: {reason}",
                "newUserStory": f"Could not generate improved user story: {reason}"
            },
            "user_centered_score": 5,
            "user_centered_justification": f"Could not analyze: {reason}",
            "user_centered_recommendations": [f"Try again: {reason}"],

            "independent_score": 5,
            "independent_justification": f"Could not analyze: {reason}",
            "independent_recommendations": [f"Try again: {reason}"],

            "negotiable_score": 5,
            "negotiable_justification": f"Could not analyze: {reason}",
            "negotiable_recommendations": [f"Try again: {reason}"],

            "valuable_score": 5,
            "valuable_justification": f"Could not analyze: {reason}",
            "valuable_recommendations": [f"Try again: {reason}"],

            "estimable_score": 5,
            "estimable_justification": f"Could not analyze: {reason}",
            "estimable_recommendations": [f"Try again: {reason}"],

            "small_score": 5,
            "small_justification": f"Could not analyze: {reason}",
            "small_recommendations": [f"Try again: {reason}"],

            "testable_score": 5,
            "testable_justification": f"Could not analyze: {reason}",
            "testable_recommendations": [f"Try again: {reason}"],

            "acceptance_criteria_score": 5,
            "acceptance_criteria_justification": f"Could not analyze: {reason}",
            "acceptance_criteria_recommendations": [f"Try again: {reason}"],

            "prioritized_score": 5,
            "prioritized_justification": f"Could not analyze: {reason}",
            "prioritized_recommendations": [f"Try again: {reason}"],

            "collaboration_score": 5,
            "collaboration_justification": f"Could not analyze: {reason}",
            "collaboration_recommendations": [f"Try again: {reason}"],

            "key_insights": [f"Analysis failed: {reason}"],
            "potential_risks": [f"Analysis failed: {reason}"],
            "recommendations": [f"Try again or check the LLM configuration: {reason}"],
            "context_analysis": {
                "most_useful_file_types": ["unknown"],
                "context_quality_assessment": f"Could not analyze context: {reason}"
            }
        }

    def _ensure_required_fields(self, analysis):
        """Ensure all required fields are present in the analysis"""
        required_fields = [
            # New recommendation fields
            ("recommendation", {
                "recommended": "no",
                "descriptionDiff": "No changes recommended",
                "acceptanceCriteriaDiff": "No changes recommended",
                "newUserStory": "Same as original user story"
            }),

            # Original fields
            ("user_centered_score", 5),
            ("user_centered_justification", "No justification provided"),
            ("user_centered_recommendations", ["No recommendations provided"]),

            ("independent_score", 5),
            ("independent_justification", "No justification provided"),
            ("independent_recommendations", ["No recommendations provided"]),

            ("negotiable_score", 5),
            ("negotiable_justification", "No justification provided"),
            ("negotiable_recommendations", ["No recommendations provided"]),

            ("valuable_score", 5),
            ("valuable_justification", "No justification provided"),
            ("valuable_recommendations", ["No recommendations provided"]),

            ("estimable_score", 5),
            ("estimable_justification", "No justification provided"),
            ("estimable_recommendations", ["No recommendations provided"]),

            ("small_score", 5),
            ("small_justification", "No justification provided"),
            ("small_recommendations", ["No recommendations provided"]),

            ("testable_score", 5),
            ("testable_justification", "No justification provided"),
            ("testable_recommendations", ["No recommendations provided"]),

            ("acceptance_criteria_score", 5),
            ("acceptance_criteria_justification", "No justification provided"),
            ("acceptance_criteria_recommendations", ["No recommendations provided"]),

            ("prioritized_score", 5),
            ("prioritized_justification", "No justification provided"),
            ("prioritized_recommendations", ["No recommendations provided"]),

            ("collaboration_score", 5),
            ("collaboration_justification", "No justification provided"),
            ("collaboration_recommendations", ["No recommendations provided"]),

            ("key_insights", ["No insights provided"]),
            ("potential_risks", ["No risks identified"]),
            ("recommendations", ["No recommendations provided"]),

            # New context analysis field
            ("context_analysis", {
                "most_useful_file_types": [],
                "context_quality_assessment": "No context analysis provided"
            })
        ]

        for field, default_value in required_fields:
            if field not in analysis or analysis[field] is None:
                logging.warning(f"Missing field in analysis: {field}, using default value")
                analysis[field] = default_value

        # Ensure scores are integers
        score_fields = [
            "user_centered_score", "independent_score", "negotiable_score",
            "valuable_score", "estimable_score", "small_score",
            "testable_score", "acceptance_criteria_score",
            "prioritized_score", "collaboration_score"
        ]

        for field in score_fields:
            try:
                analysis[field] = int(float(analysis[field]))
                # Ensure scores are in range 1-10
                analysis[field] = max(1, min(10, analysis[field]))
            except (ValueError, TypeError):
                logging.warning(f"Invalid score for {field}: {analysis.get(field)}, using default value")
                analysis[field] = 5