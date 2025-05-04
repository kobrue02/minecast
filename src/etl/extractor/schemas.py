from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional, Union
from enum import Enum
from datetime import date


class MineralType(str, Enum):
    GOLD = "gold"
    SILVER = "silver"
    COPPER = "copper"
    ZINC = "zinc"
    LEAD = "lead"
    NICKEL = "nickel"
    LITHIUM = "lithium"
    URANIUM = "uranium"
    RARE_EARTH = "rare earth"
    PLATINUM = "platinum"
    PALLADIUM = "palladium"
    IRON = "iron"
    OTHER = "other"


class ResourceCategory(str, Enum):
    MEASURED = "measured"
    INDICATED = "indicated"
    INFERRED = "inferred"
    PROVEN_RESERVE = "proven reserve"
    PROBABLE_RESERVE = "probable reserve"


class DrillHoleResult(BaseModel):
    hole_id: str = Field(..., description="Unique identifier for the drill hole")
    from_depth: float = Field(..., description="Starting depth of the intercept in meters")
    to_depth: float = Field(..., description="Ending depth of the intercept in meters")
    intercept_length: float = Field(..., description="Length of the intercept in meters")
    true_width: Optional[float] = Field(None, description="True width of the intercept if reported")
    mineral_type: MineralType = Field(..., description="Type of mineral reported")
    grade: float = Field(..., description="Grade of the mineral in appropriate units (g/t for gold/silver, % for base metals)")
    grade_unit: str = Field(..., description="Unit of measurement for the grade (g/t, %, etc.)")
    other_minerals: Optional[Dict[str, Dict[str, float]]] = Field(None, description="Other minerals in the intercept with their grades")
    
    @field_validator('intercept_length')
    def check_intercept_length(cls, v, values):
        if 'from_depth' in values and 'to_depth' in values:
            calculated = values['to_depth'] - values['from_depth']
            if abs(v - calculated) > 0.1:  # Allow for small rounding differences
                raise ValueError(f"Intercept length {v} doesn't match from-to difference {calculated}")
        return v


class DrillingNewsRelease(BaseModel):
    company_name: str = Field(..., description="Name of the mining company")
    release_date: str = Field(..., description="Date of the news release")
    project_name: str = Field(..., description="Name of the project")
    project_location: str = Field(..., description="Location of the project (country, state/province, etc.)")
    headline: str = Field(..., description="Headline of the news release")
    drilling_program: str = Field(..., description="Description of the drilling program")
    total_holes: Optional[int] = Field(None, description="Total number of drill holes reported")
    total_meters: Optional[float] = Field(None, description="Total meters drilled")
    key_highlights: List[str] = Field(..., description="Key highlights mentioned in the release")
    drill_results: List[DrillHoleResult] = Field(..., description="List of drill hole results")
    ceo_quote: Optional[str] = Field(None, description="Quote from the CEO or management")
    next_steps: Optional[List[str]] = Field(None, description="Planned next steps mentioned in the release")
    qualified_person: str = Field(..., description="Name of the qualified person who verified the results")


class ResourceEstimate(BaseModel):
    category: ResourceCategory = Field(..., description="Category of the resource estimate")
    tonnes: float = Field(..., description="Tonnage in millions")
    mineral_type: MineralType = Field(..., description="Primary mineral type")
    grade: float = Field(..., description="Grade of the primary mineral")
    grade_unit: str = Field(..., description="Unit of the grade (g/t, %, etc.)")
    contained_amount: float = Field(..., description="Contained amount of the mineral")
    contained_unit: str = Field(..., description="Unit of the contained amount (oz, lbs, tonnes, etc.)")
    cutoff_grade: Optional[float] = Field(None, description="Cut-off grade used for the estimate")
    additional_minerals: Optional[Dict[str, Dict[str, Union[float, str]]]] = Field(None, description="Additional minerals with their grades and contained amounts")


class ResourceUpdateRelease(BaseModel):
    company_name: str = Field(..., description="Name of the mining company")
    release_date: str = Field(..., description="Date of the news release")
    project_name: str = Field(..., description="Name of the project")
    project_location: str = Field(..., description="Location of the project")
    headline: str = Field(..., description="Headline of the news release")
    update_type: str = Field(..., description="Type of update (initial, updated, annual, etc.)")
    effective_date: str = Field(..., description="Effective date of the resource estimate")
    highlights: List[str] = Field(..., description="Key highlights of the resource update")
    resource_estimates: List[ResourceEstimate] = Field(..., description="List of resource estimates")
    previous_estimate_date: Optional[str] = Field(None, description="Date of the previous estimate if mentioned")
    percentage_change: Optional[Dict[str, float]] = Field(None, description="Percentage changes from previous estimate")
    estimation_methodology: Optional[str] = Field(None, description="Methodology used for the estimate")
    qualified_person: str = Field(..., description="Name of the qualified person responsible for the estimate")
    next_steps: Optional[List[str]] = Field(None, description="Planned next steps mentioned in the release")


class PermitStatus(str, Enum):
    APPLIED = "applied"
    IN_PROGRESS = "in progress"
    GRANTED = "granted"
    REJECTED = "rejected"
    APPEALED = "appealed"
    AMENDED = "amended"


class PermittingUpdate(BaseModel):
    permit_type: str = Field(..., description="Type of permit")
    status: PermitStatus = Field(..., description="Status of the permit")
    application_date: Optional[str] = Field(None, description="Date when the permit was applied for")
    expected_decision_date: Optional[str] = Field(None, description="Expected date for a decision")
    actual_decision_date: Optional[str] = Field(None, description="Actual date when a decision was made")
    issuing_authority: str = Field(..., description="Authority issuing the permit")
    details: Optional[str] = Field(None, description="Additional details about the permit")


class ProjectStatus(str, Enum):
    EXPLORATION = "exploration"
    PEA = "preliminary economic assessment"
    PFS = "pre-feasibility study"
    FS = "feasibility study"
    PERMITTING = "permitting"
    CONSTRUCTION = "construction"
    COMMISSIONING = "commissioning"
    PRODUCTION = "production"
    EXPANSION = "expansion"
    RECLAMATION = "reclamation"


class ProjectUpdateRelease(BaseModel):
    company_name: str = Field(..., description="Name of the mining company")
    release_date: str = Field(..., description="Date of the news release")
    project_name: str = Field(..., description="Name of the project")
    project_location: str = Field(..., description="Location of the project")
    headline: str = Field(..., description="Headline of the news release")
    current_status: ProjectStatus = Field(..., description="Current status of the project")
    update_type: str = Field(..., description="Type of update (milestone, delay, acceleration, etc.)")
    key_milestones: Optional[List[Dict[str, Union[str, str]]]] = Field(..., description="Key milestones with descriptions and dates")
    permitting_updates: Optional[List[PermittingUpdate]] = Field(None, description="Updates on permitting process")
    construction_progress: Optional[float] = Field(None, description="Percentage of construction completed if applicable")
    capital_expenditure_update: Optional[Dict[str, Union[float, str]]] = Field(None, description="Update on capital expenditures")
    timeline_changes: Optional[Dict[str, Union[str, str]]] = Field(None, description="Changes to project timeline")
    challenges_mentioned: Optional[List[str]] = Field(None, description="Challenges mentioned in the release")
    ceo_quote: Optional[str] = Field(None, description="Quote from the CEO or management")
    next_steps: Optional[List[str]] = Field(None, description="Planned next steps mentioned in the release")


class ManagementChangeType(str, Enum):
    APPOINTMENT = "appointment"
    RESIGNATION = "resignation"
    RETIREMENT = "retirement"
    PROMOTION = "promotion"


class ManagementChange(BaseModel):
    change_type: ManagementChangeType = Field(..., description="Type of management change")
    person_name: str = Field(..., description="Name of the person")
    position: str = Field(..., description="Position title")
    effective_date: str = Field(..., description="Effective date of the change")
    previous_position: Optional[str] = Field(None, description="Previous position if relevant")
    replacement_name: Optional[str] = Field(None, description="Name of replacement if mentioned")
    background: Optional[str] = Field(None, description="Background information about the person")


class CorporateUpdateRelease(BaseModel):
    company_name: str = Field(..., description="Name of the mining company")
    release_date: str = Field(..., description="Date of the news release")
    headline: str = Field(..., description="Headline of the news release")
    update_type: str = Field(..., description="Type of corporate update")
    management_changes: Optional[List[ManagementChange]] = Field(None, description="Management changes announced")
    financing_details: Optional[Dict[str, Union[float, str, str]]] = Field(None, description="Details of financing if announced")
    acquisition_details: Optional[Dict[str, Union[str, float, str]]] = Field(None, description="Details of acquisition if announced")
    divestiture_details: Optional[Dict[str, Union[str, float, str]]] = Field(None, description="Details of divestiture if announced")
    joint_venture_details: Optional[Dict[str, Union[str, float, str]]] = Field(None, description="Details of joint venture if announced")
    corporate_restructuring: Optional[str] = Field(None, description="Details of corporate restructuring if announced")
    key_highlights: List[str] = Field(..., description="Key highlights of the corporate update")
    ceo_quote: Optional[str] = Field(None, description="Quote from the CEO or management")


class Corp10K(BaseModel):
    company_name: str = Field(..., description="Name of the mining company")
    report_date: str = Field(..., description="Date of the 10-K report")
    earnings_per_share: float = Field(..., description="Earnings per share in USD")
    fiscal_year_end: str = Field(..., description="End date of the fiscal year")
    total_assets: float = Field(..., description="Total assets in USD")
    total_liabilities: float = Field(..., description="Total liabilities in USD")
    revenue: float = Field(..., description="Revenue in USD")
    net_income: float = Field(..., description="Net income in USD")
    earnings_per_share: float = Field(..., description="Earnings per share in USD")
    cash_and_equivalents: float = Field(..., description="Cash and cash equivalents in USD")
    total_long_term_debt: float = Field(..., description="Total long term debt in USD")

class PFSRelease(BaseModel):
    company_name: str = Field(..., description="Name of the mining company")
    release_date: str = Field(..., description="Effective Date of the news release")
    project_name: str = Field(..., description="Name of the project")
    project_location: str = Field(..., description="Location of the project")
    post_tax_net_present_value: float = Field(..., description="The Base-Case Post-tax Net Present Value (NPV)")
    long_term_gold_price_per_ounce_base_case: float = Field(..., description="The long-term gold price assumed for the base case post-tax NPV (Net Present Value) calculation")
    all_in_sustaining_cost_per_ounce: float = Field(..., description="The average all-in sustaining cost (“AISC”) per oz. over the life of mine")
    average_annual_gold_production_in_ounces: float = Field(..., description="Average annual gold production in ounces over the life of mine in commercial production")
    average_annual_free_cash_flow: float = Field(..., description="Average annual free cash flow over the life of mine in commercial production")
    average_diluted_head_grade: float = Field(..., description="Average diluted head grade in g/t")
    total_life_of_mine: float = Field(..., description="Total mine life in years")
    commercial_production_start_date: str = Field(..., description="The expected date of commercial production")
    #mineral_reserve_indicated: float = Field(..., description="Mineral reserve indicated in ounces")
    #mineral_reserve_inferred: float = Field(..., description="Mineral reserve inferred in ounces")

    