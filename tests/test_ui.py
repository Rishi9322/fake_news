import pytest
from playwright.sync_api import Page, expect

# Make sure Flask is running locally on 5000 before running this.
BASE_URL = "http://localhost:5000"

def test_landing_page(page: Page):
    """Test that the cinematic landing page loads properly."""
    page.goto(BASE_URL)
    
    # Check title
    expect(page).to_have_title("Truth Guard | AI-Based Fake News Detection")
    
    # Check main hero element
    hero_title = page.locator("h1.hero-title")
    expect(hero_title).to_be_visible()
    expect(hero_title).to_contain_text("Can You Tell What's Real Anymore?")

def test_detector_page(page: Page):
    """Test that the main detector app page loads and elements exist."""
    page.goto(f"{BASE_URL}/detector")
    
    # Check navbar brand
    expect(page.locator(".navbar-brand-custom")).to_contain_text("TRUTHGUARD")
    
    # Check that input area is available
    expect(page.locator("#newsInput")).to_be_visible()
    
    # Check analyze button
    expect(page.locator("#analyzeBtn")).to_be_visible()
    
    # Check mode toggles
    expect(page.locator("#mlModeBtn")).to_be_visible()
    expect(page.locator("#aiModeBtn")).to_be_visible()

def test_word_count_updates(page: Page):
    """Test that the word counter updates dynamically when typing."""
    page.goto(f"{BASE_URL}/detector")
    
    input_area = page.locator("#newsInput")
    word_count = page.locator("#wordCount")
    
    expect(word_count).to_contain_text("0 words")
    
    input_area.fill("This is a test sentence with seven words.")
    expect(word_count).to_contain_text("8 words")

def test_empty_submission_shows_error(page: Page):
    """Test that clicking analyze with empty text shows the error alert."""
    page.goto(f"{BASE_URL}/detector")
    
    # Click analyze without typing
    page.locator("#analyzeBtn").click()
    
    error_alert = page.locator("#errorAlert")
    expect(error_alert).to_be_visible()
    expect(error_alert).to_contain_text("Please paste a news article to analyze.")

def test_navigation_back_to_home(page: Page):
    """Test that the Back to Home button works."""
    page.goto(f"{BASE_URL}/detector")
    
    # Click back to home link
    page.locator(".btn-back").click()
    
    # Verify we are on the landing page
    expect(page).to_have_title("Truth Guard | AI-Based Fake News Detection")
