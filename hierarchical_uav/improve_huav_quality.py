"""
Improve H-UAV Prediction Quality

Strategies to improve H-UAV's 11.8% accuracy:
1. Better prompts for aerial vehicle recognition
2. Confidence-based answer filtering
3. Multi-view consistency checking
4. Answer post-processing and validation
"""

import torch
from typing import Dict, Any, Optional
import re


class HUAVQualityImprover:
    """
    Improves H-UAV prediction quality through multiple strategies.
    """

    def __init__(self):
        self.vehicle_knowledge = self._load_vehicle_knowledge()

    def _load_vehicle_knowledge(self) -> Dict[str, Any]:
        """
        Load domain knowledge about vehicles for validation.
        """
        return {
            'doors': {
                'valid_values': [2, 3, 4, 5],
                'common': [4, 2],  # Most common values
                'range': (2, 5)
            },
            'seats': {
                'valid_values': [2, 4, 5, 6, 7, 8, 9],
                'common': [5, 4, 7],
                'range': (2, 9)
            },
            'brand': {
                'valid_values': [
                    'toyota', 'honda', 'bmw', 'mercedes', 'audi',
                    'volkswagen', 'ford', 'chevrolet', 'nissan', 'hyundai'
                ]
            }
        }

    def improve_prompt(self, question: str, qtype: str, bbox_info: Dict) -> str:
        """
        Create better prompts for aerial vehicle recognition.

        Args:
            question: Original question
            qtype: Question type (doors, seats, brand, etc.)
            bbox_info: Bounding box and 3D dimensions

        Returns:
            Improved prompt
        """
        # Extract key info
        altitude = bbox_info.get('AGL', '20')
        angle = bbox_info.get('pitch_angle', '60')
        length = bbox_info.get('length', 4500)
        width = bbox_info.get('width', 1800)

        # Determine vehicle size class
        if length > 5000:
            size_class = "large vehicle (SUV, van, or truck)"
        elif length > 4200:
            size_class = "mid-size vehicle (sedan or crossover)"
        else:
            size_class = "compact vehicle (small sedan or hatchback)"

        # Build context-aware prompt
        if qtype == 'doors':
            prompt = (
                f"This is an aerial image taken from {altitude}m altitude at {angle}° angle. "
                f"The highlighted vehicle is a {size_class} with dimensions "
                f"approximately {length/1000:.1f}m × {width/1000:.1f}m. "
                f"Based on the vehicle's size and shape, how many doors does it have? "
                f"Common values are 2 (coupe), 4 (sedan), or 5 (hatchback/SUV). "
                f"Answer with just the number."
            )

        elif qtype == 'seats':
            prompt = (
                f"This is an aerial image taken from {altitude}m altitude at {angle}° angle. "
                f"The highlighted vehicle is a {size_class}. "
                f"Based on its size ({length/1000:.1f}m long), estimate the seating capacity. "
                f"Typical values: compact cars have 4-5 seats, mid-size have 5 seats, "
                f"large SUVs/vans have 6-8 seats. "
                f"Answer with just the number."
            )

        elif qtype == 'brand':
            prompt = (
                f"This is an aerial image of a vehicle taken from {altitude}m altitude. "
                f"Look at the vehicle's shape, size, and any visible markings. "
                f"What brand/manufacturer is this vehicle? "
                f"Answer with just the brand name."
            )

        else:
            # Use original question with added context
            prompt = (
                f"This is an aerial vehicle image from {altitude}m altitude at {angle}° angle. "
                f"{question} Be specific and concise."
            )

        return prompt

    def validate_answer(
        self,
        answer: str,
        qtype: str,
        confidence_score: float = None
    ) -> Dict[str, Any]:
        """
        Validate and potentially correct H-UAV's answer.

        Args:
            answer: Raw answer from H-UAV
            qtype: Question type
            confidence_score: Optional confidence from H-UAV

        Returns:
            Dictionary with:
                - validated_answer: Corrected answer if needed
                - confidence: Adjusted confidence
                - is_valid: Whether answer passes validation
                - correction_applied: What correction was made
        """
        result = {
            'validated_answer': answer,
            'confidence': confidence_score or 0.5,
            'is_valid': True,
            'correction_applied': None
        }

        # Extract numeric value for doors/seats
        if qtype in ['doors', 'seats']:
            num = self._extract_number(answer)

            if num is None:
                result['is_valid'] = False
                result['correction_applied'] = 'no_number_found'
                return result

            # Check if value is valid
            valid_values = self.vehicle_knowledge[qtype]['valid_values']
            valid_range = self.vehicle_knowledge[qtype]['range']

            if num not in valid_values:
                # Try to map to nearest valid value
                if num < valid_range[0]:
                    corrected = valid_range[0]
                elif num > valid_range[1]:
                    corrected = valid_range[1]
                else:
                    # Find nearest valid value
                    corrected = min(valid_values, key=lambda x: abs(x - num))

                result['validated_answer'] = str(int(corrected))
                result['confidence'] *= 0.7  # Reduce confidence for corrected answers
                result['correction_applied'] = f'mapped_{num}_to_{corrected}'

        # Brand validation
        elif qtype == 'brand':
            brand_lower = answer.lower().strip()
            valid_brands = self.vehicle_knowledge['brand']['valid_values']

            # Check if brand is in known list
            if brand_lower not in valid_brands:
                # Try fuzzy matching
                from difflib import get_close_matches
                matches = get_close_matches(brand_lower, valid_brands, n=1, cutoff=0.7)

                if matches:
                    result['validated_answer'] = matches[0]
                    result['confidence'] *= 0.8
                    result['correction_applied'] = f'fuzzy_match_{brand_lower}_to_{matches[0]}'
                else:
                    result['is_valid'] = False
                    result['correction_applied'] = 'unknown_brand'

        return result

    def filter_by_confidence(
        self,
        answer: str,
        self_match_score: float,
        min_threshold: float = 0.6
    ) -> Optional[str]:
        """
        Filter answers based on confidence threshold.

        Args:
            answer: H-UAV's answer
            self_match_score: Self-matching score (relevance)
            min_threshold: Minimum confidence to accept

        Returns:
            Answer if confidence is high enough, None otherwise
        """
        if self_match_score < min_threshold:
            return None  # Reject low-confidence answers

        return answer

    def apply_consistency_check(
        self,
        answer: str,
        qtype: str,
        related_answers: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Check consistency with related answers.

        For example:
        - If doors=5, seats should be >= 4 (hatchback/SUV)
        - If seats=7-9, doors should be 4-5 (large vehicle)
        - Brand should be consistent with model

        Args:
            answer: Current answer
            qtype: Question type
            related_answers: Other answers for the same vehicle

        Returns:
            Consistency check result
        """
        result = {
            'is_consistent': True,
            'conflicts': [],
            'suggested_correction': None
        }

        # Doors-Seats consistency
        if qtype == 'doors' and 'seats' in related_answers:
            doors = self._extract_number(answer)
            seats = self._extract_number(related_answers['seats'])

            if doors and seats:
                # 5 doors typically means 5+ seats
                if doors == 5 and seats < 4:
                    result['is_consistent'] = False
                    result['conflicts'].append(
                        f"5 doors usually indicates >= 4 seats, but got {seats}"
                    )

                # 2 doors typically means 2-4 seats
                if doors == 2 and seats > 5:
                    result['is_consistent'] = False
                    result['conflicts'].append(
                        f"2 doors usually indicates <= 5 seats, but got {seats}"
                    )

        return result

    @staticmethod
    def _extract_number(text):
        """Extract numeric value from text."""
        if isinstance(text, (int, float)):
            return float(text)
        match = re.search(r'\b(\d+)\b', str(text))
        if match:
            return float(match.group(1))
        return None


def apply_quality_improvements(
    huav_answer: str,
    question: str,
    qtype: str,
    self_match_score: float,
    bbox_info: Dict,
    related_answers: Dict = None
) -> Dict[str, Any]:
    """
    Apply all quality improvement strategies.

    Args:
        huav_answer: Raw answer from H-UAV
        question: Original question
        qtype: Question type
        self_match_score: Relevance score
        bbox_info: Bounding box and dimensions
        related_answers: Other answers for same vehicle

    Returns:
        Improved answer with quality metrics
    """
    improver = HUAVQualityImprover()

    # Step 1: Validate answer
    validation = improver.validate_answer(huav_answer, qtype, self_match_score)

    if not validation['is_valid']:
        return {
            'answer': None,
            'confidence': 0.0,
            'reason': 'failed_validation',
            'details': validation
        }

    # Step 2: Filter by confidence
    if self_match_score < 0.6:
        return {
            'answer': None,
            'confidence': self_match_score,
            'reason': 'low_confidence',
            'details': {'threshold': 0.6, 'score': self_match_score}
        }

    # Step 3: Consistency check
    if related_answers:
        consistency = improver.apply_consistency_check(
            validation['validated_answer'],
            qtype,
            related_answers
        )

        if not consistency['is_consistent']:
            validation['confidence'] *= 0.5  # Reduce confidence for inconsistent answers

    # Return improved answer
    return {
        'answer': validation['validated_answer'],
        'confidence': validation['confidence'],
        'reason': 'accepted',
        'details': {
            'validation': validation,
            'self_match_score': self_match_score
        }
    }


if __name__ == "__main__":
    # Test the quality improver
    improver = HUAVQualityImprover()

    # Test improved prompts
    bbox_info = {
        'AGL': '20',
        'pitch_angle': '60',
        'length': 4500,
        'width': 1800
    }

    print("Testing improved prompts:")
    print("\n1. Doors question:")
    prompt = improver.improve_prompt(
        "How many doors?",
        "doors",
        bbox_info
    )
    print(prompt)

    print("\n2. Seats question:")
    prompt = improver.improve_prompt(
        "How many seats?",
        "seats",
        bbox_info
    )
    print(prompt)

    # Test answer validation
    print("\n\nTesting answer validation:")

    # Valid answer
    result = improver.validate_answer("4", "doors")
    print(f"Answer '4' for doors: {result}")

    # Invalid answer (needs correction)
    result = improver.validate_answer("3", "doors")
    print(f"Answer '3' for doors: {result}")

    # Test consistency
    print("\n\nTesting consistency:")
    result = improver.apply_consistency_check(
        "5",  # 5 doors
        "doors",
        {'seats': '3'}  # Only 3 seats - inconsistent!
    )
    print(f"Consistency check: {result}")
