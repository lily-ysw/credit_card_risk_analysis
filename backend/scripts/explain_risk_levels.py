#!/usr/bin/env python3
"""
Explain Risk Level Determination and Additional Keywords

This script explains how risk levels were determined and shows
additional keywords that could be considered.
"""

def explain_risk_determination():
    """Explain how risk levels were determined."""
    
    print("=== HOW RISK LEVELS WERE DETERMINED ===\n")
    
    print("The risk levels were determined based on cybersecurity and fraud intelligence knowledge:")
    print("of what different terms indicate in terms of potential harm and criminal activity.\n")
    
    print("CRITICAL RISK - Why these terms are critical:")
    print("- 'fullz': Complete identity package (SSN, DOB, address, etc.) - enables identity theft")
    print("- 'dumps': Magnetic stripe data from physical cards - enables card cloning")
    print("- 'track1/track2': Raw magnetic stripe data - enables ATM fraud")
    print("- 'cashout': Instructions for converting stolen data to cash - active fraud")
    print("- 'hack/crack/exploit': Tools for breaking into systems - cyber attacks")
    print("- 'breach/leak': Stolen data from security breaches - ongoing criminal activity")
    print("- 'cloned': Duplicated cards - active fraud tool")
    print("- 'atm/bank transfer': Methods to steal money directly - financial crime")
    print()
    
    print("HIGH RISK - Why these terms are high risk:")
    print("- 'cvv': Card verification value - enables online fraud")
    print("- 'visa/mastercard/amex': Specific card types - enables targeted fraud")
    print("- 'ssn/dob': Personal identifiers - enables identity theft")
    print("- 'paypal/western union': Payment methods - enables money laundering")
    print("- 'account/bank': Financial accounts - enables account takeover")
    print("- 'pin': Personal identification numbers - enables ATM fraud")
    print("- 'balance/limit': Account information - shows financial value")
    print("- 'premium/platinum/gold': High-value cards - bigger fraud potential")
    print()
    
    print("MEDIUM RISK - Why these terms are medium risk:")
    print("- 'gift card/prepaid': Often used for money laundering")
    print("- 'method/tutorial/guide': Instructions for fraud - enables others")
    print("- 'ebook/manual': Educational materials for fraud")
    print("- 'how to/instruction': Step-by-step fraud guides")
    print()
    
    print("LOW RISK - Default when no suspicious terms found")
    print()

def show_additional_keywords():
    """Show additional keywords that could be considered."""
    
    print("=== ADDITIONAL KEYWORDS WE COULD CONSIDER ===\n")
    
    print("CRITICAL RISK - Additional terms:")
    additional_critical = [
        'skimmer', 'shimmer', 'carder', 'carding', 'dumpster', 'bins', 'cvv2', 'cvc2',
        'swiper', 'msr', 'encoder', 'embosser', 'jailbreak', 'root', 'bypass', 'exploit',
        'malware', 'trojan', 'virus', 'keylogger', 'phishing', 'spoof', 'fake', 'counterfeit',
        'money laundering', 'wash', 'clean', 'anonymous', 'stealth', 'undetectable'
    ]
    for term in additional_critical:
        print(f"  - '{term}'")
    print()
    
    print("HIGH RISK - Additional terms:")
    additional_high = [
        'bitcoin', 'crypto', 'ethereum', 'wallet', 'exchange', 'mixer', 'tumbler',
        'escrow', 'verified', 'legit', 'fresh', 'live', 'working', 'tested',
        'premium', 'vip', 'exclusive', 'private', 'invite', 'membership',
        'driver license', 'passport', 'id card', 'utility bill', 'bank statement',
        'credit report', 'background check', 'social media', 'email access'
    ]
    for term in additional_high:
        print(f"  - '{term}'")
    print()
    
    print("MEDIUM RISK - Additional terms:")
    additional_medium = [
        'discount', 'sale', 'offer', 'deal', 'bargain', 'wholesale', 'bulk',
        'reseller', 'dropshipper', 'affiliate', 'referral', 'commission',
        'software', 'tool', 'script', 'bot', 'automation', 'macro',
        'forum', 'community', 'group', 'channel', 'telegram', 'discord'
    ]
    for term in additional_medium:
        print(f"  - '{term}'")
    print()

def show_risk_assessment_logic():
    """Show the logic behind risk assessment."""
    
    print("=== RISK ASSESSMENT LOGIC ===\n")
    
    print("The risk levels are determined by asking these questions:")
    print()
    print("1. Does this enable DIRECT financial theft? (CRITICAL)")
    print("   - ATM withdrawals, bank transfers, card cloning")
    print("   - Tools for breaking into systems")
    print("   - Complete identity packages")
    print()
    print("2. Does this enable FRAUD or IDENTITY THEFT? (HIGH)")
    print("   - Credit card data, personal information")
    print("   - Payment method access")
    print("   - Account credentials")
    print()
    print("3. Does this ENABLE OTHERS to commit fraud? (MEDIUM)")
    print("   - Tutorials, guides, methods")
    print("   - Educational materials")
    print("   - Money laundering tools")
    print()
    print("4. Is this just a regular product? (LOW)")
    print("   - No suspicious terms found")
    print()

def show_examples_from_site_data():
    """Show examples of how this applies to the actual site data."""
    
    print("=== EXAMPLES FROM YOUR SITE DATA ===\n")
    
    print("CRITICAL examples found:")
    print("- '1500$ Balanced Visa Gold DUMP + FULLZ | USA'")
    print("  → Contains 'dump' (magnetic data) + 'fullz' (complete identity)")
    print("  → Enables both card cloning AND identity theft")
    print()
    print("- '1000$ – 1500$ balanced visa cloned + 1-2 day FedEx shipping'")
    print("  → Contains 'cloned' (duplicated cards)")
    print("  → Active fraud tool for ATM withdrawals")
    print()
    
    print("HIGH examples found:")
    print("- '10 x cards with credit from 1000 to 5000 USD'")
    print("  → Contains 'cards' + 'credit' + dollar amounts")
    print("  → Enables online fraud with real financial value")
    print()
    print("- 'paypal_transfer', 'paypal_gift_card'")
    print("  → Contains 'paypal' (payment method)")
    print("  → Enables money laundering and fraud")
    print()

if __name__ == "__main__":
    explain_risk_determination()
    print("=" * 60)
    show_risk_assessment_logic()
    print("=" * 60)
    show_examples_from_site_data()
    print("=" * 60)
    show_additional_keywords() 