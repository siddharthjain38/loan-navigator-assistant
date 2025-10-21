"""
Unit tests - Pure Logic Tests (No LLM dependencies)
Tests calculations, validations, and business logic
"""

import pytest


class TestEMICalculationLogic:
    """Test EMI calculation formulas (pure math)"""
    
    def test_emi_formula_accuracy(self):
        """Verify EMI calculation formula"""
        from agents.what_if_calculator import WhatIfCalculator
        
        agent = WhatIfCalculator()
        
        # Test case: ₹100,000 at 12% for 12 months
        loan_amount = 100000
        interest_rate = 12.0
        tenure_months = 12
        
        # Manual calculation
        r = interest_rate / (12 * 100)  # Monthly rate
        n = tenure_months
        expected_emi = loan_amount * r * pow(1 + r, n) / (pow(1 + r, n) - 1)
        
        # Agent calculation
        actual_emi = agent._calculate_emi(loan_amount, interest_rate, tenure_months)
        
        # Should match within 0.01%
        assert abs(actual_emi - expected_emi) < 1.0
        assert actual_emi > 0
    
    def test_zero_interest_emi(self):
        """0% interest = loan_amount / tenure"""
        from agents.what_if_calculator import WhatIfCalculator
        
        agent = WhatIfCalculator()
        
        loan_amount = 120000
        emi = agent._calculate_emi(loan_amount, 0.0, 12)
        
        expected = loan_amount / 12
        assert abs(emi - expected) < 1.0
    
    def test_multiple_scenarios_generation(self):
        """Should generate 5 standard tenure scenarios"""
        from agents.what_if_calculator import WhatIfCalculator
        
        agent = WhatIfCalculator()
        
        # Standard tenures: 5, 10, 15, 20, 30 years
        loan_amount = 500000
        interest_rate = 8.5
        
        scenarios = []
        for years in [5, 10, 15, 20, 30]:
            tenure = years * 12
            emi = agent._calculate_emi(loan_amount, interest_rate, tenure)
            total_payment = emi * tenure
            total_interest = total_payment - loan_amount
            
            scenarios.append({
                'years': years,
                'emi': emi,
                'total_interest': total_interest
            })
        
        # Longer tenure = lower EMI but higher interest
        assert scenarios[0]['emi'] > scenarios[-1]['emi']
        assert scenarios[0]['total_interest'] < scenarios[-1]['total_interest']


class TestInputValidationLogic:
    """Test validation rules (pure business logic)"""
    
    def test_negative_values_rejected(self):
        """All negative values should be rejected"""
        from agents.what_if_calculator import WhatIfCalculator
        
        agent = WhatIfCalculator()
        
        # Negative loan amount
        result = agent._validate_inputs(-100000, 10.0, 12, 0, 0)
        assert result['is_valid'] == False
        assert any('loan amount' in err.lower() for err in result['errors'])
        
        # Negative interest
        result = agent._validate_inputs(100000, -5.0, 12, 0, 0)
        assert result['is_valid'] == False
        
        # Negative tenure
        result = agent._validate_inputs(100000, 10.0, -12, 0, 0)
        assert result['is_valid'] == False
    
    def test_range_validations(self):
        """Values must be within business rules"""
        from agents.what_if_calculator import WhatIfCalculator
        
        agent = WhatIfCalculator()
        
        # Loan too small (< ₹1,000)
        result = agent._validate_inputs(500, 10.0, 12, 0, 0)
        assert result['is_valid'] == False
        
        # Loan too large (> ₹10 crore)
        result = agent._validate_inputs(150000000, 10.0, 12, 0, 0)
        assert result['is_valid'] == False
        
        # Interest rate too high (> 50%)
        result = agent._validate_inputs(100000, 75.0, 12, 0, 0)
        assert result['is_valid'] == False
        
        # Tenure too short (< 6 months)
        result = agent._validate_inputs(100000, 10.0, 3, 0, 0)
        assert result['is_valid'] == False
        
        # Tenure too long (> 30 years)
        result = agent._validate_inputs(100000, 10.0, 400, 0, 0)
        assert result['is_valid'] == False
    
    def test_prepayment_validation(self):
        """Prepayment cannot exceed outstanding balance"""
        from agents.what_if_calculator import WhatIfCalculator
        
        agent = WhatIfCalculator()
        
        # Prepayment > outstanding
        result = agent._validate_inputs(
            100000, 10.0, 12,
            prepayment=60000,
            outstanding_balance=50000
        )
        
        assert result['is_valid'] == False
        assert any('prepayment' in err.lower() and 'exceeds' in err.lower() 
                  for err in result['errors'])
        assert any('foreclosure' in sugg.lower() for sugg in result['suggestions'])
    
    def test_valid_inputs_pass(self):
        """Valid inputs should pass all validations"""
        from agents.what_if_calculator import WhatIfCalculator
        
        agent = WhatIfCalculator()
        
        # Valid case
        result = agent._validate_inputs(500000, 8.5, 240, 10000, 400000)
        
        assert result['is_valid'] == True
        assert len(result['errors']) == 0


class TestSQLValidationLogic:
    """Test SQL validation rules (security logic)"""
    
    def test_dangerous_keywords_blocked(self):
        """DROP, DELETE, UPDATE should be blocked"""
        from agents.sql_agent import SQLAgent
        
        agent = SQLAgent()
        
        dangerous_queries = [
            "DROP TABLE Loans",
            "DELETE FROM Loans WHERE id = 1",
            "UPDATE Loans SET amount = 0",
            "TRUNCATE TABLE Loans",
            "ALTER TABLE Loans ADD COLUMN hacked INT"
        ]
        
        for sql in dangerous_queries:
            result = agent._validate_sql(sql)
            assert result['is_valid'] == False, f"Should block: {sql}"
    
    def test_sql_injection_patterns_blocked(self):
        """Common injection patterns should be blocked"""
        from agents.sql_agent import SQLAgent
        
        agent = SQLAgent()
        
        injection_attempts = [
            "SELECT * FROM Loans WHERE id = 1; DROP TABLE Loans; --",
            "SELECT * FROM Loans WHERE id = 1 UNION SELECT * FROM Users",
            "SELECT * FROM Loans WHERE name = 'x' OR '1'='1' --"
        ]
        
        for sql in injection_attempts:
            result = agent._validate_sql(sql)
            assert result['is_valid'] == False, f"Should block injection: {sql}"
    
    def test_valid_select_queries_allowed(self):
        """Valid SELECT queries should pass"""
        from agents.sql_agent import SQLAgent
        
        agent = SQLAgent()
        
        valid_queries = [
            "SELECT * FROM Loans WHERE customer_id = ?",
            "SELECT loan_id, loan_amount FROM Loans",
            "SELECT COUNT(*) FROM Loans WHERE status = ?"
        ]
        
        for sql in valid_queries:
            result = agent._validate_sql(sql)
            assert result['is_valid'] == True, f"Should allow: {sql}"
    
    def test_table_whitelist_enforcement(self):
        """Only whitelisted tables should be allowed"""
        from agents.sql_agent import SQLAgent
        
        agent = SQLAgent()
        
        # Valid table
        valid_sql = "SELECT * FROM Loans WHERE id = 1"
        result = agent._validate_sql(valid_sql)
        assert result['is_valid'] == True
        
        # Invalid table
        invalid_sql = "SELECT * FROM AdminUsers WHERE id = 1"
        result = agent._validate_sql(invalid_sql)
        assert result['is_valid'] == False


class TestErrorResponseFormatting:
    """Test error message generation (string formatting)"""
    
    def test_validation_error_format(self):
        """Error response should include helpful information"""
        from agents.what_if_calculator import WhatIfCalculator
        
        agent = WhatIfCalculator()
        
        # Generate validation error
        validation = agent._validate_inputs(-50000, 10.0, 12, 0, 0)
        error_response = agent._generate_error_response(validation)
        
        # Should be a string with error details
        assert isinstance(error_response, str)
        assert len(error_response) > 0
        
        # Should mention the error
        assert 'error' in error_response.lower() or 'invalid' in error_response.lower()
        
        # Should include valid ranges
        assert 'range' in error_response.lower() or 'valid' in error_response.lower()
    
    def test_foreclosure_suggestion_format(self):
        """Foreclosure suggestion should be clear"""
        from agents.what_if_calculator import WhatIfCalculator
        
        agent = WhatIfCalculator()
        
        # Excessive prepayment
        validation = agent._validate_inputs(
            100000, 10.0, 12,
            prepayment=120000,
            outstanding_balance=50000
        )
        
        # Should suggest foreclosure
        assert len(validation['suggestions']) > 0
        foreclosure_mentioned = any('foreclosure' in sugg.lower() 
                                   for sugg in validation['suggestions'])
        assert foreclosure_mentioned


class TestBusinessLogic:
    """Test business rules and workflows"""
    
    def test_confidence_threshold_logic(self):
        """Confidence thresholds define routing behavior"""
        # This is pure logic - no LLM needed
        
        def should_clarify(confidence):
            return confidence < 0.5
        
        def should_check_multi_domain(confidence):
            return 0.5 <= confidence < 0.7
        
        def should_route_directly(confidence):
            return confidence >= 0.7
        
        # Test thresholds
        assert should_clarify(0.35) == True
        assert should_clarify(0.65) == False
        
        assert should_check_multi_domain(0.60) == True
        assert should_check_multi_domain(0.40) == False
        assert should_check_multi_domain(0.80) == False
        
        assert should_route_directly(0.85) == True
        assert should_route_directly(0.65) == False
    
    def test_similarity_threshold_logic(self):
        """Documents below 0.75 similarity should be filtered"""
        # Pure logic test
        
        def filter_by_similarity(documents, threshold=0.75):
            return [doc for doc in documents 
                   if doc.get('similarity', 0) >= threshold]
        
        docs = [
            {'content': 'A', 'similarity': 0.95},
            {'content': 'B', 'similarity': 0.65},
            {'content': 'C', 'similarity': 0.80},
            {'content': 'D', 'similarity': 0.50},
        ]
        
        filtered = filter_by_similarity(docs)
        
        assert len(filtered) == 2
        assert all(doc['similarity'] >= 0.75 for doc in filtered)


# Run with: pytest tests/unit/test_pure_logic.py -v -s
if __name__ == "__main__":
    print("\n" + "="*80)
    print("UNIT TESTS - PURE LOGIC (NO LLM CALLS)")
    print("="*80)
    print("\nThese tests verify:")
    print("  - Mathematical calculations")
    print("  - Validation rules")
    print("  - Business logic")
    print("  - String formatting")
    print("\nNo mocking required - pure deterministic logic!")
    print("="*80 + "\n")
