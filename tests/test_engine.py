import asyncio
import json
import types

from src.engine import (
    Engine,
    _guess_hours,
    _guess_pay,
    _guess_remote,
    _heuristic_opportunity,
    _parse_ddg_html,
    _search_angles,
)
from src.models import Opportunity


# --- compensation from listing text (never invented) --------------------


def test_guess_pay_parses_real_numbers_and_refuses_to_invent():
    assert _guess_pay("Senior ML Engineer", "$180k") == 180_000
    assert _guess_pay("Staff Engineer $150,000", "") == 150_000
    assert _guess_pay("Engineer", "$120k-$180k") == 180_000
    assert _guess_pay("Engineer", "$143,000 to 197,000") == 197_000
    assert _guess_pay("Engineer", "USD 200,000–240,000") == 240_000
    from src.engine import _parse_pay
    assert _parse_pay("**Salary:** USD 160,000–190,000") == (160_000, 190_000)
    assert _parse_pay("Base Salary: $126,000 - $180,000Diversity") == (126_000, 180_000)
    assert _parse_pay("proposed band b/t US$175k and $250k annually") == (175_000, 250_000)
    assert _parse_pay("$160,000 and $190,000") == (160_000, 190_000)
    assert _parse_pay("$180,000 and $5,000 signing bonus") == (None, 180_000)
    assert _parse_pay("Salary range: $190,000 $250,000 + performance-based bonus") == (
        190_000,
        250_000,
    )
    assert _parse_pay("$180K $200K") == (180_000, 200_000)
    assert _parse_pay("Salary: $157-200kApplicants must be authorized") == (
        157_000,
        200_000,
    )
    assert _parse_pay("Base Pay Range: $160,000 USD - $240,000 USD") == (
        160_000,
        240_000,
    )
    assert _guess_pay("Software Engineer", "") is None
    assert _guess_pay("Senior Staff Principal Lead", "junior intern") is None
    assert _parse_pay("without $500K comp") == (None, None)
    assert _parse_pay(
        "Top performing radiologists can expect to earn up to $950,000+."
    ) == (None, None)
    assert _parse_pay("Salary $180k without $500K bonus") == (None, 180_000)
    assert _parse_pay("Compensation up to $180,000") == (None, 180_000)
    assert _parse_pay("$150,000 in equity") == (None, None)
    assert _parse_pay("equity: $150,000") == (None, None)
    assert _parse_pay("RSUs of $80,000") == (None, None)
    assert _parse_pay("$150k-$200k equity") == (None, None)
    assert _parse_pay("Salary $180,000 plus $50,000 equity") == (None, 180_000)
    assert _parse_pay("$180k plus $50k in RSUs") == (None, 180_000)
    assert _parse_pay("$180,000 a year") == (None, 180_000)
    assert _parse_pay("$20,000 signing bonus") == (None, None)
    assert _parse_pay("signing bonus of $25,000") == (None, None)
    assert _parse_pay("$10,000 relocation bonus") == (None, None)
    assert _parse_pay("relocation bonus of $10,000") == (None, None)
    assert _parse_pay("$10,000 relocation") == (None, None)
    assert _parse_pay("$10,000 relocation assistance") == (None, None)
    assert _parse_pay("relocation of $10,000") == (None, None)
    assert _parse_pay("Salary $180,000 plus $10,000 relocation") == (None, 180_000)
    assert _parse_pay("$180,000 relocation to Seattle") == (None, 180_000)
    assert _parse_pay("$15,000 tuition reimbursement") == (None, None)
    assert _parse_pay("$20,000 education benefit") == (None, None)
    assert _parse_pay("$15,000 tuition assistance") == (None, None)
    assert _parse_pay("$15,000 education reimbursement") == (None, None)
    assert _parse_pay("$15,000 tuition") == (None, None)
    assert _parse_pay("tuition reimbursement of $15,000") == (None, None)
    assert _parse_pay("education reimbursement of $15,000") == (None, None)
    assert _parse_pay("$10,000 student loan repayment") == (None, None)
    assert _parse_pay("Salary $180,000 plus $15,000 tuition reimbursement") == (
        None,
        180_000,
    )
    assert _parse_pay("$180,000 tuition in NYC") == (None, 180_000)
    assert _parse_pay("$25,000 professional development budget") == (None, None)
    assert _parse_pay("$10,000 professional development") == (None, None)
    assert _parse_pay("$10,000 learning budget") == (None, None)
    assert _parse_pay("$10,000 learning and development budget") == (None, None)
    assert _parse_pay("$10,000 learning and development") == (None, None)
    assert _parse_pay("Salary $180,000 learning opportunities") == (None, 180_000)
    assert _parse_pay("$10,000 continuing education") == (None, None)
    assert _parse_pay("$10,000 education budget") == (None, None)
    assert _parse_pay("professional development budget of $25,000") == (None, None)
    assert _parse_pay("learning budget of $10,000") == (None, None)
    assert _parse_pay("Salary $180,000 plus $25,000 professional development budget") == (
        None,
        180_000,
    )
    assert _parse_pay("$180,000 professional in NYC") == (None, 180_000)
    assert _parse_pay("$15,000 per month") == (None, 180_000)
    assert _parse_pay("$10,000 conference budget") == (None, None)
    assert _parse_pay("$10,000 training reimbursement") == (None, None)
    assert _parse_pay("$10,000 training budget") == (None, None)
    assert _parse_pay("$10,000 wellness benefit") == (None, None)
    assert _parse_pay("$15,000 parental leave") == (None, None)
    assert _parse_pay("$10,000 fertility benefit") == (None, None)
    assert _parse_pay("$10,000 childcare benefit") == (None, None)
    assert _parse_pay("$10,000 child care benefit") == (None, None)
    assert _parse_pay("$10,000 dependent care benefit") == (None, None)
    assert _parse_pay("$10,000 childcare stipend") == (None, None)
    assert _parse_pay("childcare benefit of $10,000") == (None, None)
    assert _parse_pay("Salary $180,000 plus $10,000 childcare benefit") == (
        None,
        180_000,
    )
    assert _parse_pay("$180,000 childcare in NYC") == (None, 180_000)
    assert _parse_pay("conference budget of $10,000") == (None, None)
    assert _parse_pay("parental leave of $15,000") == (None, None)
    assert _parse_pay("Salary $180,000 plus $10,000 conference budget") == (
        None,
        180_000,
    )
    assert _parse_pay("$180,000 training in NYC") == (None, 180_000)
    assert _parse_pay("$15,000 per month") == (None, 180_000)
    assert _parse_pay("$25,000 annual bonus") == (None, None)
    assert _parse_pay("target bonus of $25,000") == (None, None)
    assert _parse_pay("$25,000 performance bonus") == (None, None)
    assert _parse_pay("$25,000 retention bonus") == (None, None)
    assert _parse_pay("Salary $180,000 plus $20,000 signing bonus") == (None, 180_000)
    assert _parse_pay("$25,000 sign-on") == (None, None)
    assert _parse_pay("$25,000 sign on") == (None, None)
    assert _parse_pay("sign-on of $25,000") == (None, None)
    assert _parse_pay("sign-on: $25,000") == (None, None)
    assert _parse_pay("$15,000 referral bonus") == (None, None)
    assert _parse_pay("$50,000 employee referral bonus") == (None, None)
    assert _parse_pay("referral bonus of $15,000") == (None, None)
    assert _parse_pay("$10,000 spot bonus") == (None, None)
    assert _parse_pay("spot bonus of $10,000") == (None, None)
    assert _parse_pay("$12,000 wellness stipend") == (None, None)
    assert _parse_pay("$10,000 cell phone stipend") == (None, None)
    assert _parse_pay("$10,000 phone stipend") == (None, None)
    assert _parse_pay("$10,000 internet stipend") == (None, None)
    assert _parse_pay("$10,000 commuter stipend") == (None, None)
    assert _parse_pay("$10,000 home office stipend") == (None, None)
    assert _parse_pay("$10,000 gym stipend") == (None, None)
    assert _parse_pay("$10,000 gym membership stipend") == (None, None)
    assert _parse_pay("$10,000 fitness stipend") == (None, None)
    assert _parse_pay("$10,000 gym allowance") == (None, None)
    assert _parse_pay("$10,000 fitness reimbursement") == (None, None)
    assert _parse_pay("$10,000 gym reimbursement") == (None, None)
    assert _parse_pay("$10,000 commuter reimbursement") == (None, None)
    assert _parse_pay("$10,000 parking reimbursement") == (None, None)
    assert _parse_pay("$10,000 phone reimbursement") == (None, None)
    assert _parse_pay("$15,000 home office reimbursement") == (None, None)
    assert _parse_pay("fitness reimbursement of $10,000") == (None, None)
    assert _parse_pay("Salary $180,000 plus $10,000 fitness reimbursement") == (
        None,
        180_000,
    )
    assert _parse_pay("$180,000 fitness in NYC") == (None, 180_000)
    assert _parse_pay("$10,000 commuter benefit") == (None, None)
    assert _parse_pay("$10,000 parking benefit") == (None, None)
    assert _parse_pay("$10,000 phone benefit") == (None, None)
    assert _parse_pay("$10,000 internet benefit") == (None, None)
    assert _parse_pay("commuter benefit of $10,000") == (None, None)
    assert _parse_pay("Salary $180,000 plus $10,000 commuter benefit") == (
        None,
        180_000,
    )
    assert _parse_pay("$180,000 commuter in NYC") == (None, 180_000)
    assert _parse_pay("$180,000 phone in NYC") == (None, 180_000)
    assert _parse_pay("Salary $180,000 plus $10,000 gym stipend") == (
        None,
        180_000,
    )
    assert _parse_pay("$180,000 gym in NYC") == (None, 180_000)
    assert _parse_pay("$10,000 gym membership") == (None, None)
    assert _parse_pay("$10,000 fitness membership") == (None, None)
    assert _parse_pay("gym membership of $10,000") == (None, None)
    assert _parse_pay("Salary $180,000 plus $10,000 gym membership") == (
        None,
        180_000,
    )
    assert _parse_pay("$10,000 annual wellness") == (None, None)
    assert _parse_pay("annual wellness of $10,000") == (None, None)
    assert _parse_pay("$10,000 mental health benefit") == (None, None)
    assert _parse_pay("mental health benefit of $10,000") == (None, None)
    assert _parse_pay("$10,000 transit benefit") == (None, None)
    assert _parse_pay("transit benefit of $10,000") == (None, None)
    assert _parse_pay("Salary $180,000 plus $10,000 transit benefit") == (
        None,
        180_000,
    )
    assert _parse_pay("$180,000 transit in NYC") == (None, 180_000)
    assert _parse_pay("$10,000 HSA contribution") == (None, None)
    assert _parse_pay("$10,000 FSA contribution") == (None, None)
    assert _parse_pay("HSA contribution of $10,000") == (None, None)
    assert _parse_pay("Salary $180,000 plus $10,000 HSA contribution") == (
        None,
        180_000,
    )
    assert _parse_pay("$180,000 HSA in NYC") == (None, 180_000)
    assert _parse_pay("$10,000 healthcare stipend") == (None, None)
    assert _parse_pay("$10,000 health stipend") == (None, None)
    assert _parse_pay("Salary $180,000 plus $10,000 healthcare stipend") == (
        None,
        180_000,
    )
    assert _parse_pay("$180,000 health in NYC") == (None, 180_000)
    assert _parse_pay("$10,000 medical benefit") == (None, None)
    assert _parse_pay("$10,000 dental benefit") == (None, None)
    assert _parse_pay("$10,000 vision benefit") == (None, None)
    assert _parse_pay("medical benefit of $10,000") == (None, None)
    assert _parse_pay("Salary $180,000 plus $10,000 medical benefit") == (
        None,
        180_000,
    )
    assert _parse_pay("$180,000 medical in NYC") == (None, 180_000)
    assert _parse_pay("$10,000 ESPP") == (None, None)
    assert _parse_pay("$10,000 employee stock purchase") == (None, None)
    assert _parse_pay("ESPP of $10,000") == (None, None)
    assert _parse_pay("Salary $180,000 plus $10,000 ESPP") == (None, 180_000)
    assert _parse_pay("$10,000 annual 401k match") == (None, None)
    assert _parse_pay("$2,000 monthly cell phone") == (None, None)
    assert _parse_pay("$2,000 per month internet") == (None, None)
    assert _parse_pay("Salary $180,000 plus $2,000/month cell phone") == (None, 180_000)
    assert _parse_pay("$10,000 cash bonus") == (None, None)
    assert _parse_pay("$10,000 year-end bonus") == (None, None)
    assert _parse_pay("$10,000 holiday bonus") == (None, None)
    assert _parse_pay("Salary $180,000 plus $10,000 cash bonus") == (None, 180_000)
    assert _parse_pay("$180,000 cash in NYC") == (None, 180_000)
    assert _parse_pay("$10,000 life insurance") == (None, None)
    assert _parse_pay("$10,000 disability insurance") == (None, None)
    assert _parse_pay("$10,000 short-term disability") == (None, None)
    assert _parse_pay("$180,000 life in NYC") == (None, 180_000)
    assert _parse_pay("$2,000 monthly parking") == (None, None)
    assert _parse_pay("$2,000 monthly gym") == (None, None)
    assert _parse_pay("$2,000 monthly commuter") == (None, None)
    assert _parse_pay("$2,000 monthly wellness") == (None, None)
    assert _parse_pay("Salary $180,000 plus $2,000 monthly parking") == (None, 180_000)
    assert _parse_pay("$10,000 matching 401k") == (None, None)
    assert _parse_pay("$10,000 401k matching") == (None, None)
    assert _parse_pay("$10,000 dependent care FSA") == (None, None)
    assert _parse_pay("$10,000 legal insurance") == (None, None)
    assert _parse_pay("$10,000 LTD benefit") == (None, None)
    assert _parse_pay("$180,000 LTD in NYC") == (None, 180_000)
    assert _parse_pay("$10,000 PTO buyback") == (None, None)
    assert _parse_pay("$10,000 PTO cashout") == (None, None)
    assert _parse_pay("PTO buyback of $10,000") == (None, None)
    assert _parse_pay("$180,000 PTO in NYC") == (None, 180_000)
    assert _parse_pay("$10,000 identity theft protection") == (None, None)
    assert _parse_pay("$10,000 severance") == (None, None)
    assert _parse_pay("severance of $10,000") == (None, None)
    assert _parse_pay("Salary $180,000 plus $10,000 severance") == (None, 180_000)
    assert _parse_pay("$10,000 discretionary bonus") == (None, None)
    assert _parse_pay("$10,000 quarterly bonus") == (None, None)
    assert _parse_pay("$10,000 monthly bonus") == (None, None)
    assert _parse_pay("$10,000 incentive bonus") == (None, None)
    assert _parse_pay("$10,000 sales bonus") == (None, None)
    assert _parse_pay("$10,000 stay bonus") == (None, None)
    assert _parse_pay("$10,000 anniversary bonus") == (None, None)
    assert _parse_pay("$20,000 bonus") == (None, 20_000)
    assert _parse_pay("Salary $180,000 plus $10,000 discretionary bonus") == (
        None,
        180_000,
    )
    assert _parse_pay("$10,000 pet insurance") == (None, None)
    assert _parse_pay("$10,000 vision insurance") == (None, None)
    assert _parse_pay("$10,000 medical insurance") == (None, None)
    assert _parse_pay("pet insurance of $10,000") == (None, None)
    assert _parse_pay("Salary $180,000 plus $10,000 pet insurance") == (None, 180_000)
    assert _parse_pay("$180,000 medical in NYC") == (None, 180_000)
    assert _parse_pay("$10,000 pet") == (None, 10_000)
    assert _parse_pay("$10,000 legal plan") == (None, None)
    assert _parse_pay("$10,000 legal") == (None, 10_000)
    assert _parse_pay("$10,000 adoption assistance") == (None, None)
    assert _parse_pay("$10,000 HRA contribution") == (None, None)
    assert _parse_pay("$10,000 HRA") == (None, 10_000)
    assert _parse_pay("$10,000 WFH stipend") == (None, None)
    assert _parse_pay("$10,000 work from home stipend") == (None, None)
    assert _parse_pay("$10,000 equipment stipend") == (None, None)
    assert _parse_pay("$10,000 laptop stipend") == (None, None)
    assert _parse_pay("$10,000 vacation payout") == (None, None)
    assert _parse_pay("vacation buyback of $10,000") == (None, None)
    assert _parse_pay("$10,000 unused PTO") == (None, None)
    assert _parse_pay("$10,000 family leave") == (None, None)
    assert _parse_pay("$10,000 clothing allowance") == (None, None)
    assert _parse_pay("$10,000 hiring bonus") == (None, None)
    assert _parse_pay("$10,000 welcome bonus") == (None, None)
    assert _parse_pay("$10,000 tech stipend") == (None, None)
    assert _parse_pay("$10,000 transit stipend") == (None, None)
    assert _parse_pay("$10,000 WFH allowance") == (None, None)
    assert _parse_pay("$10,000 sick payout") == (None, None)
    assert _parse_pay("$10,000 critical illness insurance") == (None, None)
    assert _parse_pay("$10,000 legal benefit") == (None, None)
    assert _parse_pay("$10,000 baby bonus") == (None, None)
    assert _parse_pay("$10,000 peer bonus") == (None, None)
    assert _parse_pay("$10,000 referral award") == (None, None)
    assert _parse_pay("$10,000 employee referral") == (None, None)
    assert _parse_pay("$10,000 mileage reimbursement") == (None, None)
    assert _parse_pay("$10,000 gym benefit") == (None, None)
    assert _parse_pay("$10,000 STD benefit") == (None, None)
    assert _parse_pay("$10,000 profit share") == (None, None)
    assert _parse_pay("$10,000 restricted stock") == (None, None)
    assert _parse_pay("$10,000 employee stock") == (None, None)
    assert _parse_pay("$10,000 pension contribution") == (None, None)
    assert _parse_pay("$10,000 529 contribution") == (None, None)
    assert _parse_pay("$10,000 matching gift") == (None, None)
    assert _parse_pay("$10,000 charitable match") == (None, None)
    assert _parse_pay("$10,000 spot award") == (None, None)
    assert _parse_pay("$10,000 holiday gift") == (None, None)
    assert _parse_pay("$10,000 incentive compensation") == (None, None)
    assert _parse_pay("$10,000 variable pay") == (None, None)
    assert _parse_pay("$10,000 variable compensation") == (None, None)
    assert _parse_pay("$10,000 deferred compensation") == (None, None)
    assert _parse_pay("$10,000 QSEHRA contribution") == (None, None)
    assert _parse_pay("$10,000 ICHRA contribution") == (None, None)
    assert _parse_pay("$10,000 COBRA subsidy") == (None, None)
    assert _parse_pay("$10,000 clothing stipend") == (None, None)
    assert _parse_pay("$10,000 long-term incentive") == (None, None)
    assert _parse_pay("$10,000 option grant") == (None, None)
    assert _parse_pay("$10,000 variable bonus") == (None, None)
    assert _parse_pay("$10,000 cell phone reimbursement") == (None, None)
    assert _parse_pay("$10,000 donation match") == (None, None)
    assert _parse_pay("$10,000 pet benefit") == (None, None)
    assert _parse_pay("$10,000 cellphone allowance") == (None, None)
    assert _parse_pay("$10,000 fitness benefit") == (None, None)
    assert _parse_pay("$10,000 mileage stipend") == (None, None)
    assert _parse_pay("$10,000 caregiver allowance") == (None, None)
    assert _parse_pay("$10,000 pension") == (None, 10_000)
    assert _parse_pay("$20,000 bonus") == (None, 20_000)
    assert _parse_pay("$180,000 gym in NYC") == (None, 180_000)
    assert _parse_pay("$10,000 wellness") == (None, 10_000)
    assert _parse_pay("$10,000 HSA") == (None, 10_000)
    assert _parse_pay("$10,000 monthly internet stipend") == (None, None)
    assert _parse_pay("$10,000 cell phone allowance") == (None, None)
    assert _parse_pay("Salary $180,000 plus $10,000 cell phone stipend") == (
        None,
        180_000,
    )
    assert _parse_pay("$15,000 per month") == (None, 180_000)
    assert _parse_pay("Salary $180,000 plus $50,000 employee referral bonus") == (
        None,
        180_000,
    )
    assert _parse_pay("Salary $180,000 plus $25,000 sign-on") == (None, 180_000)
    assert _parse_pay("Salary $180,000 plus $25,000 annual bonus") == (None, 180_000)
    assert _parse_pay("$2,000/month stipend") == (None, None)
    assert _parse_pay("$2,000 monthly stipend") == (None, None)
    assert _parse_pay("$2,000/month housing stipend") == (None, None)
    assert _parse_pay("$2,000 housing stipend") == (None, None)
    assert _parse_pay("stipend of $2,000 per month") == (None, None)
    assert _parse_pay("Salary $180,000 plus $2,000/month stipend") == (None, 180_000)
    assert _parse_pay("Salary $180,000 plus $2,000/month housing stipend") == (
        None,
        180_000,
    )
    assert _parse_pay("$2,000/month housing allowance") == (None, None)
    assert _parse_pay("$2,000/month housing") == (None, None)
    assert _parse_pay("Housing: $2,000/month") == (None, None)
    assert _parse_pay("Housing $2,000 per month") == (None, None)
    assert _parse_pay("$15,000 per month housing") == (None, None)
    assert _parse_pay("housing of $2,000 per month") == (None, None)
    assert _parse_pay("Salary $15,000 per month. Housing not provided") == (
        None,
        180_000,
    )
    assert _parse_pay("$3,000/month car allowance") == (None, None)
    assert _parse_pay("housing allowance of $2,000 per month") == (None, None)
    assert _parse_pay("car allowance of $3,000 per month") == (None, None)
    assert _parse_pay("Salary $180,000 plus $3,000/month car allowance") == (
        None,
        180_000,
    )
    assert _parse_pay("$15,000 per month") == (None, 180_000)
    assert _parse_pay("$250,000 OTE") == (None, None)
    assert _parse_pay("OTE $250,000") == (None, None)
    assert _parse_pay("on-target earnings of $250,000") == (None, None)
    assert _parse_pay("$180,000-$250,000 OTE") == (None, None)
    assert _parse_pay("Base $180,000. OTE $250,000") == (None, 180_000)
    assert _parse_pay("$80,000 commission") == (None, None)
    assert _parse_pay("commission of $80,000") == (None, None)
    assert _parse_pay("$200,000 total compensation") == (None, None)
    assert _parse_pay("total compensation of $200,000") == (None, None)
    assert _parse_pay("$200k TC") == (None, None)
    assert _parse_pay("TC: $200,000") == (None, None)
    assert _parse_pay("Base $180,000. TC $250,000") == (None, 180_000)
    assert _parse_pay("$10,000 401(k) match") == (None, None)
    assert _parse_pay("$12,000 employer 401k match") == (None, None)
    assert _parse_pay("401(k) match of $10,000") == (None, None)
    assert _parse_pay("$15,000 profit sharing") == (None, None)
    assert _parse_pay("profit-sharing of $15,000") == (None, None)
    assert _parse_pay("Salary $180,000 plus $10,000 401(k) match") == (None, 180_000)
    assert _parse_pay("Salary $180,000 plus $15,000 profit sharing") == (None, 180_000)
    assert _parse_pay("$180,000") == (None, 180_000)
    assert _parse_pay("$80/hr overtime") == (None, None)
    assert _parse_pay("Overtime paid at $80/hr") == (None, None)
    assert _parse_pay("$25/hr on-call") == (None, None)
    assert _parse_pay("$5/hr shift differential") == (None, None)
    assert _parse_pay("Salary $180,000. Overtime at $80/hr") == (None, 180_000)
    assert _parse_pay("$80/hr") == (None, 160_000)


_SIGNIFYD_GEO_PAY = """
Tier 1 (NYC/SF Bay Area/Seattle): $160,000 - $190,000
Tier 2 (DC Metro/Austin/Boston/Los Angeles): $150,000 - $180,000
Tier 3 (US - All Other): $140,000 - $170,000
"""


def test_parse_pay_prefers_remote_geo_band():
    from src.engine import _parse_pay, _remote_geo_pay

    assert _parse_pay(_SIGNIFYD_GEO_PAY) == (160_000, 190_000)
    assert _parse_pay(_SIGNIFYD_GEO_PAY, remote=False) == (160_000, 190_000)
    assert _parse_pay(_SIGNIFYD_GEO_PAY, remote=True) == (140_000, 170_000)
    assert _remote_geo_pay(_SIGNIFYD_GEO_PAY) == (140_000, 170_000)
    assert _parse_pay("Tier 3 (US - All Other): $140k - $170k", remote=True) == (
        140_000,
        170_000,
    )
    assert _parse_pay(
        "NYC: $160,000 - $190,000\nRemote: $140,000 - $170,000", remote=True
    ) == (140_000, 170_000)
    assert _parse_pay(
        "We're a remote company. Salary: $160,000 - $190,000", remote=True
    ) == (160_000, 190_000)
    assert _parse_pay("$80 - $100 / Hour", remote=True) == (160_000, 200_000)


def test_foreign_salary_detects_k_suffix_gbp_and_eur():
    from src.engine import _foreign_salary, _parse_pay

    for blob in ("£60k", "£60K - £80K", "€85k", "GBP 60k", "EUR 85k"):
        assert _parse_pay(blob) == (None, None)
        assert _foreign_salary(f"<p>{blob} a year</p>") is True
    assert _foreign_salary("<p>$60k a year</p>") is False
    assert _foreign_salary("<p>Apply now. No salary listed.</p>") is False


def test_foreign_salary_detects_mxn_cad_and_salario_dollars():
    from src.engine import _foreign_salary, _parse_pay

    mx = "Salario bruto mensual entre $20,000 y $25,000"
    assert _parse_pay(mx) == (None, None)
    assert _foreign_salary(f"<p>{mx}</p>") is True
    assert _parse_pay("CAD $160,000 - $180,000") == (None, None)
    assert _foreign_salary("<p>CAD $160,000 - $180,000</p>") is True
    assert _parse_pay("C$90,000") == (None, None)
    assert _foreign_salary("<p>Pay is $180,000 CAD a year</p>") is True
    assert _parse_pay("$160,000 - 200,000 (CAD)") == (None, None)
    assert _foreign_salary("<p>The salary range for this role is $160,000 - 200,000 (CAD)</p>") is True
    assert _parse_pay("$196,000—$269,500 CAD") == (None, None)
    assert _foreign_salary("<p>Annual Base Salary$196,000—$269,500 CAD</p>") is True
    assert _parse_pay("$180,000 a year") == (None, 180_000)
    assert _foreign_salary("<p>$180,000 a year</p>") is False
    mixed = "UK £45,000 – £60,000. US $240,000 - $500,000"
    assert _parse_pay(mixed) == (None, None)
    assert _foreign_salary(f"<p>{mixed}</p>") is True
    chf = "The salary is CHF 150,000. US equivalent $180,000"
    assert _parse_pay(chf) == (None, None)
    assert _foreign_salary(f"<p>{chf}</p>") is True
    assert _parse_pay("CHF 91'052") == (None, None)
    assert _foreign_salary("<p>CHF 91'052</p>") is True
    assert _parse_pay("Compensation: 150,000 CHF") == (None, None)
    assert _foreign_salary("<p>Compensation: 150,000 CHF</p>") is True
    assert _parse_pay("INR 2,400,000. US $180,000") == (None, None)
    assert _foreign_salary("<p>₹12,00,000 or $180,000</p>") is True
    assert _parse_pay("SEK 800,000. US equivalent $90,000") == (None, None)
    assert _foreign_salary("<p>SEK 800,000. US equivalent $90,000</p>") is True
    assert _parse_pay("Compensation: 750,000 NOK") == (None, None)
    assert _foreign_salary("<p>Compensation: 750,000 NOK</p>") is True
    assert _parse_pay("PLN 240,000 or $180,000") == (None, None)
    assert _parse_pay("$15000 to $17000 gross Salary Monthly") == (None, None)
    assert _foreign_salary("<p>$15000 to $17000 gross Salary Monthly</p>") is True


def test_foreign_salary_detects_kr_and_zl_without_iso_code():
    from src.engine import _apply_listing, _foreign_salary, _parse_pay

    kr = "800 000 kr. US equivalent $90,000"
    assert _parse_pay(kr) == (None, None)
    assert _foreign_salary(f"<p>{kr}</p>") is True
    assert _parse_pay("800.000 kr") == (None, None)
    assert _foreign_salary("<p>Compensation: 750000 kr</p>") is True
    assert _parse_pay("kr 750 000") == (None, None)
    assert _parse_pay("800 000:-") == (None, None)
    zl = "240 000 zł or $180,000"
    assert _parse_pay(zl) == (None, None)
    assert _foreign_salary(f"<p>{zl}</p>") is True
    assert _parse_pay("$180,000 a year") == (None, 180_000)
    assert _foreign_salary("<p>$180,000 a year</p>") is False
    opp = Opportunity(title="Engineer", url="https://jobs.example/se")
    listed = _apply_listing(opp, f"<p>{kr}</p>")
    assert listed is False
    assert opp.pay_high is None


def test_foreign_salary_detects_prefixed_dollars_and_rs():
    from src.engine import _apply_listing, _foreign_salary, _parse_pay

    assert _parse_pay("R$ 180,000") == (None, None)
    assert _foreign_salary("<p>R$ 180,000 a year</p>") is True
    assert _parse_pay("R$180,000. US equivalent $40,000") == (None, None)
    assert _parse_pay("HK$ 180,000") == (None, None)
    assert _foreign_salary("<p>HK$ 180,000 a year</p>") is True
    assert _parse_pay("S$ 12,000") == (None, None)
    assert _parse_pay("NZ$ 140,000") == (None, None)
    assert _parse_pay("Rs. 12,00,000 or $90,000") == (None, None)
    assert _foreign_salary("<p>Rs. 15,00,000 a year</p>") is True
    assert _parse_pay("Rs 2400000") == (None, None)
    assert _parse_pay("15 LPA. US equivalent $90,000") == (None, None)
    assert _foreign_salary("<p>15-20 LPA. US equivalent $90,000</p>") is True
    assert _parse_pay("CTC 18 lakhs") == (None, None)
    assert _parse_pay("US$ 180,000") == (None, 180_000)
    assert _parse_pay("$180,000 a year") == (None, 180_000)
    assert _foreign_salary("<p>$180,000 a year</p>") is False
    opp = Opportunity(title="Engineer", url="https://jobs.example/br")
    listed = _apply_listing(opp, "<p>R$ 180,000 a year. US equivalent $40,000</p>")
    assert listed is False
    assert opp.pay_high is None
    lpa = Opportunity(title="Engineer", url="https://jobs.example/in")
    listed = _apply_listing(lpa, "<p>15 LPA. US equivalent $90,000</p>")
    assert listed is False
    assert lpa.pay_high is None


def test_guess_pay_annualizes_hourly():
    assert _guess_pay("Contract", "$80/hr") == 160_000  # 80 * 40 * 50
    assert _guess_pay("Contract", "$80/hr", hours=20) == 80_000
    assert _guess_pay("", "$80 - $100 / Hour") == 200_000
    from src.engine import _parse_pay
    assert _parse_pay("$80 - $100 / Hour") == (160_000, 200_000)
    assert _parse_pay("$80–$100/hr") == (160_000, 200_000)
    assert _parse_pay("$80 an hour") == (None, 160_000)
    assert _parse_pay("$80 hourly") == (None, 160_000)
    assert _parse_pay("$80 an hr") == (None, 160_000)
    assert _parse_pay("USD 80 per hour") == (None, 160_000)
    assert _parse_pay("$80–$100 an hour") == (160_000, 200_000)
    assert _parse_pay("$180,000 a year") == (None, 180_000)
    assert _parse_pay("Salary $180,000 plus $80/hr on-call") == (None, 180_000)
    assert _parse_pay("Salary $180,000 plus $800/day travel") == (None, 180_000)
    assert _parse_pay("$800/day travel") == (None, None)
    assert _parse_pay("travel of $800 per day") == (None, None)
    assert _parse_pay("Travel: $800/day") == (None, None)
    assert _parse_pay("$3,000 per week travel") == (None, None)
    assert _parse_pay("Salary $180,000 plus $3,000 per week travel") == (None, 180_000)
    assert _parse_pay("Base $200,000. $15,000 per month housing") == (None, 200_000)
    assert _parse_pay("$30/hour meal") == (None, None)
    assert _parse_pay("$50/day meal") == (None, None)
    assert _parse_pay("$75/day food") == (None, None)
    assert _parse_pay("$2,000/month living") == (None, None)
    assert _parse_pay("Meal: $30/hour") == (None, None)
    assert _parse_pay("Food: $75/day") == (None, None)
    assert _parse_pay("Living: $2,000/month") == (None, None)
    assert _parse_pay("meal of $50 per day") == (None, None)
    assert _parse_pay("$50/day for meals") == (None, None)
    assert _parse_pay("Salary $180,000 plus $30/hour meal") == (None, 180_000)
    assert _parse_pay("$180,000 meal in NYC") == (None, 180_000)
    assert _parse_pay("$100 per diem") == (None, 25_000)
    assert _parse_pay("$15,000 per month") == (None, 180_000)
    assert _parse_pay("$80/hr") == (None, 160_000)
    from src.engine import _apply_listing

    travel = Opportunity(title="Engineer", url="https://jobs.example/travel")
    assert _apply_listing(travel, "<p>$800/day travel. Great team.</p>") is False
    assert travel.pay_high is None
    paid = Opportunity(title="Engineer", url="https://jobs.example/travel-sal")
    assert _apply_listing(
        paid, "<p>Salary $180,000 plus $800/day travel</p>"
    ) is True
    assert paid.pay_high == 180_000
    meal = Opportunity(title="Engineer", url="https://jobs.example/meal")
    assert _apply_listing(meal, "<p>$30/hour meal. Great team.</p>") is False
    assert meal.pay_high is None
    meal_sal = Opportunity(title="Engineer", url="https://jobs.example/meal-sal")
    assert _apply_listing(
        meal_sal, "<p>Salary $180,000 plus $30/hour meal</p>"
    ) is True
    assert meal_sal.pay_high == 180_000
    tuition = Opportunity(title="Engineer", url="https://jobs.example/tuition")
    assert _apply_listing(
        tuition, "<p>$15,000 tuition reimbursement. Great team.</p>"
    ) is False
    assert tuition.pay_high is None
    tuition_sal = Opportunity(title="Engineer", url="https://jobs.example/tuition-sal")
    assert _apply_listing(
        tuition_sal, "<p>Salary $180,000 plus $15,000 tuition reimbursement</p>"
    ) is True
    assert tuition_sal.pay_high == 180_000
    devel = Opportunity(title="Engineer", url="https://jobs.example/lnd")
    assert _apply_listing(
        devel, "<p>$25,000 professional development budget. Great team.</p>"
    ) is False
    assert devel.pay_high is None
    devel_sal = Opportunity(title="Engineer", url="https://jobs.example/lnd-sal")
    assert _apply_listing(
        devel_sal,
        "<p>Salary $180,000 plus $25,000 professional development budget</p>",
    ) is True
    assert devel_sal.pay_high == 180_000
    conf = Opportunity(title="Engineer", url="https://jobs.example/conf")
    assert _apply_listing(
        conf, "<p>$10,000 conference budget. Great team.</p>"
    ) is False
    assert conf.pay_high is None
    leave = Opportunity(title="Engineer", url="https://jobs.example/leave")
    assert _apply_listing(
        leave, "<p>$15,000 parental leave. Great team.</p>"
    ) is False
    assert leave.pay_high is None
    care = Opportunity(title="Engineer", url="https://jobs.example/childcare")
    assert _apply_listing(
        care, "<p>$10,000 childcare benefit. Great team.</p>"
    ) is False
    assert care.pay_high is None
    care_sal = Opportunity(title="Engineer", url="https://jobs.example/childcare-sal")
    assert _apply_listing(
        care_sal, "<p>Salary $180,000 plus $10,000 childcare benefit</p>"
    ) is True
    assert care_sal.pay_high == 180_000
    phone = Opportunity(title="Engineer", url="https://jobs.example/phone")
    assert _apply_listing(
        phone, "<p>$10,000 monthly internet stipend. Great team.</p>"
    ) is False
    assert phone.pay_high is None
    phone_sal = Opportunity(title="Engineer", url="https://jobs.example/phone-sal")
    assert _apply_listing(
        phone_sal, "<p>Salary $180,000 plus $10,000 cell phone stipend</p>"
    ) is True
    assert phone_sal.pay_high == 180_000
    gym = Opportunity(title="Engineer", url="https://jobs.example/gym")
    assert _apply_listing(
        gym, "<p>$10,000 gym stipend. Great team.</p>"
    ) is False
    assert gym.pay_high is None
    gym_sal = Opportunity(title="Engineer", url="https://jobs.example/gym-sal")
    assert _apply_listing(
        gym_sal, "<p>Salary $180,000 plus $10,000 gym stipend</p>"
    ) is True
    assert gym_sal.pay_high == 180_000
    fitness = Opportunity(title="Engineer", url="https://jobs.example/fitness")
    assert _apply_listing(
        fitness, "<p>$10,000 fitness reimbursement. Great team.</p>"
    ) is False
    assert fitness.pay_high is None
    fitness_sal = Opportunity(title="Engineer", url="https://jobs.example/fitness-sal")
    assert _apply_listing(
        fitness_sal, "<p>Salary $180,000 plus $10,000 fitness reimbursement</p>"
    ) is True
    assert fitness_sal.pay_high == 180_000
    commute_ben = Opportunity(title="Engineer", url="https://jobs.example/commute-ben")
    assert _apply_listing(
        commute_ben, "<p>$10,000 commuter benefit. Great team.</p>"
    ) is False
    assert commute_ben.pay_high is None
    commute_ben_sal = Opportunity(title="Engineer", url="https://jobs.example/commute-ben-sal")
    assert _apply_listing(
        commute_ben_sal, "<p>Salary $180,000 plus $10,000 commuter benefit</p>"
    ) is True
    assert commute_ben_sal.pay_high == 180_000
    gym_mem = Opportunity(title="Engineer", url="https://jobs.example/gym-mem")
    assert _apply_listing(
        gym_mem, "<p>$10,000 gym membership. Great team.</p>"
    ) is False
    assert gym_mem.pay_high is None
    hsa = Opportunity(title="Engineer", url="https://jobs.example/hsa")
    assert _apply_listing(
        hsa, "<p>$10,000 HSA contribution. Great team.</p>"
    ) is False
    assert hsa.pay_high is None
    hsa_sal = Opportunity(title="Engineer", url="https://jobs.example/hsa-sal")
    assert _apply_listing(
        hsa_sal, "<p>Salary $180,000 plus $10,000 HSA contribution</p>"
    ) is True
    assert hsa_sal.pay_high == 180_000
    health_stip = Opportunity(title="Engineer", url="https://jobs.example/health-stip")
    assert _apply_listing(
        health_stip, "<p>$10,000 healthcare stipend. Great team.</p>"
    ) is False
    assert health_stip.pay_high is None
    medical_ben = Opportunity(title="Engineer", url="https://jobs.example/medical-ben")
    assert _apply_listing(
        medical_ben, "<p>$10,000 medical benefit. Great team.</p>"
    ) is False
    assert medical_ben.pay_high is None
    espp = Opportunity(title="Engineer", url="https://jobs.example/espp")
    assert _apply_listing(espp, "<p>$10,000 ESPP. Great team.</p>") is False
    assert espp.pay_high is None
    cell_mo = Opportunity(title="Engineer", url="https://jobs.example/cell-mo")
    assert _apply_listing(
        cell_mo, "<p>$2,000 monthly cell phone. Great team.</p>"
    ) is False
    assert cell_mo.pay_high is None
    cash_b = Opportunity(title="Engineer", url="https://jobs.example/cash-b")
    assert _apply_listing(cash_b, "<p>$10,000 cash bonus. Apply now.</p>") is False
    assert cash_b.pay_high is None
    park_mo = Opportunity(title="Engineer", url="https://jobs.example/park-mo")
    assert _apply_listing(
        park_mo, "<p>$2,000 monthly parking. Great team.</p>"
    ) is False
    assert park_mo.pay_high is None
    match401 = Opportunity(title="Engineer", url="https://jobs.example/m401")
    assert _apply_listing(match401, "<p>$10,000 matching 401k. Apply now.</p>") is False
    assert match401.pay_high is None
    pto = Opportunity(title="Engineer", url="https://jobs.example/pto")
    assert _apply_listing(pto, "<p>$10,000 PTO buyback. Great team.</p>") is False
    assert pto.pay_high is None
    sev = Opportunity(title="Engineer", url="https://jobs.example/sev")
    assert _apply_listing(sev, "<p>$10,000 severance. Apply now.</p>") is False
    assert sev.pay_high is None
    pet = Opportunity(title="Engineer", url="https://jobs.example/pet")
    assert _apply_listing(pet, "<p>$10,000 pet insurance. Great team.</p>") is False
    assert pet.pay_high is None
    wfh_stip = Opportunity(title="Engineer", url="https://jobs.example/wfh-stip")
    assert _apply_listing(
        wfh_stip, "<p>$10,000 WFH stipend. Apply now.</p>"
    ) is False
    assert wfh_stip.pay_high is None
    vac = Opportunity(title="Engineer", url="https://jobs.example/vac")
    assert _apply_listing(vac, "<p>$10,000 vacation payout. Great team.</p>") is False
    assert vac.pay_high is None
    hire_b = Opportunity(title="Engineer", url="https://jobs.example/hire-b")
    assert _apply_listing(hire_b, "<p>$10,000 hiring bonus. Apply now.</p>") is False
    assert hire_b.pay_high is None
    tech_stip = Opportunity(title="Engineer", url="https://jobs.example/tech-stip")
    assert _apply_listing(tech_stip, "<p>$10,000 tech stipend. Great team.</p>") is False
    assert tech_stip.pay_high is None
    baby_b = Opportunity(title="Engineer", url="https://jobs.example/baby-b")
    assert _apply_listing(baby_b, "<p>$10,000 baby bonus. Apply now.</p>") is False
    assert baby_b.pay_high is None
    gym_ben = Opportunity(title="Engineer", url="https://jobs.example/gym-ben")
    assert _apply_listing(gym_ben, "<p>$10,000 gym benefit. Great team.</p>") is False
    assert gym_ben.pay_high is None
    gift = Opportunity(title="Engineer", url="https://jobs.example/gift")
    assert _apply_listing(gift, "<p>$10,000 matching gift. Apply now.</p>") is False
    assert gift.pay_high is None
    emp_st = Opportunity(title="Engineer", url="https://jobs.example/emp-st")
    assert _apply_listing(emp_st, "<p>$10,000 employee stock. Great team.</p>") is False
    assert emp_st.pay_high is None
    inc_comp = Opportunity(title="Engineer", url="https://jobs.example/inc-comp")
    assert _apply_listing(
        inc_comp, "<p>$10,000 incentive compensation. Apply now.</p>"
    ) is False
    assert inc_comp.pay_high is None
    cobra = Opportunity(title="Engineer", url="https://jobs.example/cobra")
    assert _apply_listing(cobra, "<p>$10,000 COBRA subsidy. Great team.</p>") is False
    assert cobra.pay_high is None
    lti = Opportunity(title="Engineer", url="https://jobs.example/lti")
    assert _apply_listing(lti, "<p>$10,000 long-term incentive. Apply now.</p>") is False
    assert lti.pay_high is None
    don = Opportunity(title="Engineer", url="https://jobs.example/don-match")
    assert _apply_listing(don, "<p>$10,000 donation match. Great team.</p>") is False
    assert don.pay_high is None


def test_parse_pay_annualizes_monthly_usd():
    from src.engine import _apply_listing, _parse_pay

    assert _parse_pay("$15,000 per month") == (None, 180_000)
    assert _parse_pay("$15,000/month") == (None, 180_000)
    assert _parse_pay("$15,000 a month") == (None, 180_000)
    assert _parse_pay("$15k/month") == (None, 180_000)
    assert _parse_pay("USD 15,000 per month") == (None, 180_000)
    assert _parse_pay("$15,000 monthly") == (None, 180_000)
    assert _parse_pay("$8,000-$10,000 per month") == (96_000, 120_000)
    assert _parse_pay("$10,000-$15,000 per month") == (120_000, 180_000)
    assert _parse_pay("$8k–$12k/month") == (96_000, 144_000)
    assert _parse_pay("$20k-$25k/month") == (240_000, 300_000)
    assert _parse_pay("$80/hr") == (None, 160_000)
    assert _parse_pay("$180,000 a year") == (None, 180_000)
    opp = Opportunity(title="Engineer", url="https://jobs.example/mo")
    assert _apply_listing(opp, "<p>Salary $15,000 per month</p>") is True
    assert opp.pay_high == 180_000
    housing = Opportunity(title="Engineer", url="https://jobs.example/house")
    assert _apply_listing(housing, "<p>$2,000/month housing. Great team.</p>") is False
    assert housing.pay_high is None
    salary = Opportunity(title="Engineer", url="https://jobs.example/house-sal")
    assert _apply_listing(
        salary, "<p>Salary $180,000 plus $2,000/month housing</p>"
    ) is True
    assert salary.pay_high == 180_000


def test_parse_pay_annualizes_weekly_usd():
    from src.engine import _apply_listing, _parse_pay

    assert _parse_pay("$3,000 per week") == (None, 150_000)
    assert _parse_pay("$3,000/week") == (None, 150_000)
    assert _parse_pay("$3,000 a week") == (None, 150_000)
    assert _parse_pay("$3k/week") == (None, 150_000)
    assert _parse_pay("$3,000/wk") == (None, 150_000)
    assert _parse_pay("USD 3,000 per week") == (None, 150_000)
    assert _parse_pay("$3,000 weekly") == (None, 150_000)
    assert _parse_pay("$1,500-$2,000 per week") == (75_000, 100_000)
    assert _parse_pay("$2k–$4k/week") == (100_000, 200_000)
    assert _parse_pay("$80/hr") == (None, 160_000)
    assert _parse_pay("$15,000 per month") == (None, 180_000)
    assert _parse_pay("$180,000 a year") == (None, 180_000)
    opp = Opportunity(title="Engineer", url="https://jobs.example/wk")
    assert _apply_listing(opp, "<p>Salary $3,000 per week</p>") is True
    assert opp.pay_high == 150_000


def test_parse_pay_annualizes_biweekly_and_semimonthly_usd():
    from src.engine import _apply_listing, _parse_pay

    assert _parse_pay("$3,000 biweekly") == (None, 75_000)
    assert _parse_pay("$3,000 bi-weekly") == (None, 75_000)
    assert _parse_pay("$3,000 every two weeks") == (None, 75_000)
    assert _parse_pay("$3,000 fortnightly") == (None, 75_000)
    assert _parse_pay("$1,500-$2,000 biweekly") == (37_500, 50_000)
    assert _parse_pay("$3,000 twice a month") == (None, 72_000)
    assert _parse_pay("$3,000 semi-monthly") == (None, 72_000)
    assert _parse_pay("$3,000 per week") == (None, 150_000)
    assert _parse_pay("$15,000 per month") == (None, 180_000)
    opp = Opportunity(title="Engineer", url="https://jobs.example/bi")
    assert _apply_listing(opp, "<p>Pay $3,000 biweekly</p>") is True
    assert opp.pay_high == 75_000
    semi = Opportunity(title="Engineer", url="https://jobs.example/sm")
    assert _apply_listing(semi, "<p>Pay $3,000 semi-monthly</p>") is True
    assert semi.pay_high == 72_000


def test_apply_listing_json_ld_biweekly_pay():
    from src.engine import _apply_listing

    html = """
    <script type="application/ld+json">
    {"@type":"JobPosting","title":"Engineer",
     "baseSalary":{"@type":"MonetaryAmount","currency":"USD",
       "value":{"@type":"QuantitativeValue","value":3000,"unitText":"BIWEEKLY"}}}
    </script>
    """
    opp = Opportunity(title="Engineer", url="https://jobs.example/ld-bi")
    assert _apply_listing(opp, html) is True
    assert opp.pay_high == 75_000
    thousands = """
    <script type="application/ld+json">
    {"@type":"JobPosting","title":"Engineer",
     "baseSalary":{"@type":"MonetaryAmount","currency":"USD",
       "value":{"@type":"QuantitativeValue","value":3,"unitText":"BIWEEKLY"}}}
    </script>
    """
    k = Opportunity(title="Engineer", url="https://jobs.example/ld-bi-k")
    assert _apply_listing(k, thousands) is True
    assert k.pay_high == 75_000
    semi_k = """
    <script type="application/ld+json">
    {"@type":"JobPosting","title":"Engineer",
     "baseSalary":{"@type":"MonetaryAmount","currency":"USD",
       "value":{"@type":"QuantitativeValue","value":3,"unitText":"SEMI-MONTHLY"}}}
    </script>
    """
    sm = Opportunity(title="Engineer", url="https://jobs.example/ld-sm-k")
    assert _apply_listing(sm, semi_k) is True
    assert sm.pay_high == 72_000


def test_parse_pay_annualizes_daily_usd():
    from src.engine import _apply_listing, _parse_pay

    assert _parse_pay("$800/day") == (None, 200_000)
    assert _parse_pay("$800 per day") == (None, 200_000)
    assert _parse_pay("$800 a day") == (None, 200_000)
    assert _parse_pay("$800 daily") == (None, 200_000)
    assert _parse_pay("USD 800 per day") == (None, 200_000)
    assert _parse_pay("$400-$500/day") == (100_000, 125_000)
    assert _parse_pay("$400–$600 per day") == (100_000, 150_000)
    assert _parse_pay("$400 per diem") == (None, 100_000)
    assert _parse_pay("$400/diem") == (None, 100_000)
    assert _parse_pay("USD 400 per diem") == (None, 100_000)
    assert _parse_pay("$400-$500 per diem") == (100_000, 125_000)
    assert _parse_pay("$80/hr") == (None, 160_000)
    assert _parse_pay("$3,000 per week") == (None, 150_000)
    assert _parse_pay("$180,000 a year") == (None, 180_000)
    opp = Opportunity(title="Engineer", url="https://jobs.example/day")
    assert _apply_listing(opp, "<p>Rate $800 per day</p>") is True
    assert opp.pay_high == 200_000


def test_apply_listing_json_ld_daily_pay():
    from src.engine import _apply_listing

    html = """
    <script type="application/ld+json">
    {"@type":"JobPosting","title":"Engineer",
     "baseSalary":{"@type":"MonetaryAmount","currency":"USD",
       "value":{"@type":"QuantitativeValue","minValue":400,"maxValue":500,"unitText":"DAY"}}}
    </script>
    """
    opp = Opportunity(title="Engineer", url="https://jobs.example/ld-day")
    assert _apply_listing(opp, html) is True
    assert opp.pay_low == 100_000
    assert opp.pay_high == 125_000
    diem = """
    <script type="application/ld+json">
    {"@type":"JobPosting","title":"Engineer",
     "baseSalary":{"@type":"MonetaryAmount","currency":"USD",
       "value":{"@type":"QuantitativeValue","value":400,"unitText":"PER_DIEM"}}}
    </script>
    """
    per = Opportunity(title="Engineer", url="https://jobs.example/ld-diem")
    assert _apply_listing(per, diem) is True
    assert per.pay_high == 100_000
    forty = """
    <script type="application/ld+json">
    {"@type":"JobPosting","title":"Engineer",
     "baseSalary":{"@type":"MonetaryAmount","currency":"USD",
       "value":{"@type":"QuantitativeValue","value":40,"unitText":"DAY"}}}
    </script>
    """
    day_rate = Opportunity(title="Engineer", url="https://jobs.example/ld-day-40")
    assert _apply_listing(day_rate, forty) is True
    assert day_rate.pay_high == 10_000


def test_guess_pay_reads_description_not_just_title():
    assert _guess_pay("Engineer", "comp $175k plus equity") == 175_000


def test_apply_listing_does_not_rank_equity_as_salary():
    from src.engine import _apply_listing

    opp = Opportunity(title="Engineer", url="https://jobs.example/x")
    assert _apply_listing(opp, "<p>$150,000 in equity. Apply now.</p>") is False
    assert opp.pay_high is None
    paid = Opportunity(title="Engineer", url="https://jobs.example/y")
    assert _apply_listing(paid, "<p>Salary $180,000 plus $50,000 equity</p>") is True
    assert paid.pay_high == 180_000
    bonus = Opportunity(title="Engineer", url="https://jobs.example/z")
    assert _apply_listing(bonus, "<p>$20,000 signing bonus. Apply now.</p>") is False
    assert bonus.pay_high is None
    both = Opportunity(title="Engineer", url="https://jobs.example/w")
    assert _apply_listing(both, "<p>Salary $180,000 plus $20,000 signing bonus</p>") is True
    assert both.pay_high == 180_000
    annual = Opportunity(title="Engineer", url="https://jobs.example/a")
    assert _apply_listing(annual, "<p>$25,000 annual bonus. Apply now.</p>") is False
    assert annual.pay_high is None
    mixed = Opportunity(title="Engineer", url="https://jobs.example/m")
    assert _apply_listing(mixed, "<p>Salary $180,000 plus $25,000 annual bonus</p>") is True
    assert mixed.pay_high == 180_000
    ote = Opportunity(title="Account Executive", url="https://jobs.example/ote")
    assert _apply_listing(ote, "<p>$250,000 OTE. Apply now.</p>") is False
    assert ote.pay_high is None
    base = Opportunity(title="Account Executive", url="https://jobs.example/base")
    assert _apply_listing(base, "<p>Base $180,000. OTE $250,000</p>") is True
    assert base.pay_high == 180_000
    comm = Opportunity(title="Account Executive", url="https://jobs.example/comm")
    assert _apply_listing(comm, "<p>$80,000 commission. Apply now.</p>") is False
    assert comm.pay_high is None
    tc = Opportunity(title="Engineer", url="https://jobs.example/tc")
    assert _apply_listing(tc, "<p>$200k TC. Apply now.</p>") is False
    assert tc.pay_high is None
    tcmix = Opportunity(title="Engineer", url="https://jobs.example/tcmix")
    assert _apply_listing(tcmix, "<p>Base $180,000. TC $250,000</p>") is True
    assert tcmix.pay_high == 180_000
    kmatch = Opportunity(title="Engineer", url="https://jobs.example/401k")
    assert _apply_listing(kmatch, "<p>$10,000 401(k) match. Apply now.</p>") is False
    assert kmatch.pay_high is None
    profit = Opportunity(title="Engineer", url="https://jobs.example/ps")
    assert _apply_listing(profit, "<p>$15,000 profit sharing. Apply now.</p>") is False
    assert profit.pay_high is None
    kbase = Opportunity(title="Engineer", url="https://jobs.example/401base")
    assert _apply_listing(
        kbase, "<p>Salary $180,000 plus $10,000 401(k) match</p>"
    ) is True
    assert kbase.pay_high == 180_000
    ot = Opportunity(title="Engineer", url="https://jobs.example/ot")
    assert _apply_listing(ot, "<p>$80/hr overtime. Apply now.</p>") is False
    assert ot.pay_high is None
    otbase = Opportunity(title="Engineer", url="https://jobs.example/otbase")
    assert _apply_listing(
        otbase, "<p>Salary $180,000. Overtime paid at $80/hr</p>"
    ) is True
    assert otbase.pay_high == 180_000


def test_guess_hours_from_text_not_job_type():
    assert _guess_hours("Engineer", "20 hrs/week") == 20
    assert _guess_hours("Engineer", "32 hours a week") == 32
    assert _guess_hours("Engineer", "32 hours a week. This is a full-time role.") == 32
    assert _guess_hours("Engineer", "37.5 hours per week") == 38
    assert _guess_hours("Engineer", "37.5 hrs/week") == 38
    assert _guess_hours("Engineer", "40 hours weekly") == 40
    assert _guess_hours("Engineer", "50 hour work week") == 50
    assert _guess_hours("Engineer", "50-hour workweek") == 50
    assert _guess_hours("Engineer", "50 hours of work a week") == 50
    assert _guess_hours("Engineer", "50 hours of work per week") == 50
    assert _guess_hours("Engineer", "40 hours of work weekly") == 40
    assert _guess_hours("Engineer", "50-hour work week") == 50
    assert _guess_hours("Engineer", "45 hour work-week") == 45
    assert _guess_hours("Engineer", "40 hrs. per week") == 40
    assert _guess_hours("Engineer", "2 hour weekly meeting") is None
    assert _guess_hours("Engineer", "2-hour weekly standup") is None
    assert _guess_hours("Engineer", "12 weeks of parental leave") is None
    assert _guess_hours("Part-time role", "") == 20
    assert _guess_hours("Full-time Engineer", "") == 40
    assert _guess_hours("Contract Engineer", "") is None
    assert _guess_hours("Engineer", "") is None
    assert _guess_hours("Engineer", "partner teams") is None
    assert (
        _guess_hours(
            "Staff, Machine Learning Engineer",
            "education benefit program for full-time and part-time associates",
        )
        is None
    )
    assert _guess_hours("Part-time role", "full-time and part-time associates") == 20
    assert (
        _guess_hours(
            "Engineer",
            "This is a full-time role. Benefits for full-time and part-time associates.",
        )
        == 40
    )


def test_apply_listing_reads_hours_a_week_for_rate():
    from src.engine import _apply_listing

    html = "<title>Engineer at Acme</title><p>$160,000 a year. 32 hours a week.</p>"
    opp = Opportunity(title="Engineer", url="https://jobs.example/x")
    _apply_listing(opp, html)
    assert opp.pay_high == 160_000
    assert opp.hours_per_week == 32
    assert opp.rate_is_imputed is False
    assert opp.score() == 100.0
    frac = Opportunity(title="Engineer", url="https://jobs.example/frac")
    assert _apply_listing(
        frac, "<p>$180,000 a year. 37.5 hours per week.</p>"
    ) is True
    assert frac.hours_per_week == 38
    assert frac.rate_is_imputed is False
    assert frac.score() == 180_000 / (38 * 50)
    ld = """
    <script type="application/ld+json">
    {"@type":"JobPosting","title":"Engineer","workHours":"37.5 hours per week",
     "baseSalary":{"currency":"USD","value":{"value":180000,"unitText":"YEAR"}}}
    </script>
    """
    json_hours = Opportunity(title="Engineer", url="https://jobs.example/ld-hrs")
    assert _apply_listing(json_hours, ld) is True
    assert json_hours.hours_per_week == 38
    assert json_hours.score() == 180_000 / (38 * 50)
    workweek = Opportunity(title="Engineer", url="https://jobs.example/workweek")
    assert _apply_listing(
        workweek, "<p>$180,000 a year. This is a 50-hour workweek.</p>"
    ) is True
    assert workweek.hours_per_week == 50
    assert workweek.rate_is_imputed is False
    assert workweek.score() == 180_000 / (50 * 50)
    meeting = Opportunity(title="Engineer", url="https://jobs.example/meeting")
    assert _apply_listing(
        meeting, "<p>$180,000 a year. 2 hour weekly meeting.</p>"
    ) is True
    assert meeting.hours_per_week is None
    assert meeting.score() == 180_000 / (40 * 50)
    ld_week = """
    <script type="application/ld+json">
    {"@type":"JobPosting","title":"Engineer","workHours":"50-hour workweek",
     "baseSalary":{"currency":"USD","value":{"value":180000,"unitText":"YEAR"}}}
    </script>
    """
    json_week = Opportunity(title="Engineer", url="https://jobs.example/ld-workweek")
    assert _apply_listing(json_week, ld_week) is True
    assert json_week.hours_per_week == 50
    assert json_week.score() == 180_000 / (50 * 50)
    of_work = Opportunity(title="Engineer", url="https://jobs.example/of-work")
    assert _apply_listing(
        of_work, "<p>$180,000 a year. 50 hours of work a week.</p>"
    ) is True
    assert of_work.hours_per_week == 50
    assert of_work.score() == 180_000 / (50 * 50)


def test_apply_listing_benefits_boilerplate_is_not_part_time():
    from src.engine import _apply_listing

    html = (
        "<title>Staff, Machine Learning Engineer - Walmart Careers</title>"
        "<p>Bentonville, AR (onsite) $130,000 - $260,000</p>"
        "<p>education benefit program for full-time and part-time associates</p>"
    )
    opp = Opportunity(
        title="Staff, Machine Learning Engineer",
        url="https://careers.walmart.com/us/en/jobs/R-2395925",
    )
    _apply_listing(opp, html)
    assert opp.hours_per_week is None
    assert opp.rate_is_imputed is True
    assert opp.pay_low == 130_000
    assert opp.pay_high == 260_000
    assert opp.remote is False
    assert opp.score() == 91.0


def test_apply_listing_stated_hours_beat_part_time_default():
    from src.engine import _apply_listing

    html = """
    <script type="application/ld+json">
    {"@type":"JobPosting","title":"Therapist","employmentType":"PART_TIME",
     "baseSalary":{"currency":"USD","value":{"minValue":80000,"maxValue":80000,"unitText":"YEAR"}}}
    </script>
    <p>Approximately 24 hours per week.</p>
    """
    opp = Opportunity(title="Therapist", url="https://jobs.example/x")
    _apply_listing(opp, html)
    assert opp.hours_per_week == 24
    assert opp.pay_high == 80_000
    assert opp.rate_is_imputed is False
    assert opp.score() == 80_000 / (24 * 50)
    assert _guess_remote("Engineer", "hybrid schedule") is False
    assert _guess_remote("Engineer", "Must be hybrid, $80/hr") is False
    assert _guess_remote("Engineer", "must be onsite") is False
    assert _guess_remote("Engineer", "must work on site") is False
    assert _guess_remote("Engineer", "Work from our office in NYC") is False
    assert _guess_remote("Engineer", "work from our New York office") is False
    assert _guess_remote("Engineer", "work from the San Francisco office") is False
    assert _guess_remote("Engineer", "must work from our Seattle office") is False
    assert _guess_remote("Engineer", "work out of our New York office") is False
    assert _guess_remote("Engineer", "work out of the Seattle office") is False
    assert _guess_remote("Engineer", "based out of our Austin office") is False
    assert _guess_remote("Engineer", "We're based out of New York. Great team.") is True
    assert _guess_remote("Engineer", "This is an office-based role") is False
    assert _guess_remote("Engineer", "this role is office first") is False
    assert _guess_remote("Engineer", "office-first role") is False
    assert _guess_remote("Engineer", "office first role") is False
    assert _guess_remote("Engineer", "office first aid training required") is True
    assert _guess_remote("Engineer", "remote-first role") is True
    assert _guess_remote("Engineer", "This is a site-based role") is False
    assert _guess_remote("Engineer", "This is a campus-based position") is False
    assert _guess_remote("Engineer", "This is an on-campus role") is False
    assert _guess_remote("Engineer", "must work on campus") is False
    assert _guess_remote("Engineer", "work from our campus in Boston") is False
    assert _guess_remote("Engineer", "must report to our NYC office") is False
    assert _guess_remote("Engineer", "must commute to San Francisco") is False
    assert _guess_remote("Engineer", "must commute to our San Francisco office") is False
    assert _guess_remote("Engineer", "regularly commute to our Detroit office") is False
    assert _guess_remote("Engineer", "must commute to the US") is True
    assert _guess_remote("Engineer", "must commute to interviews") is True
    assert _guess_remote("Engineer", "on-campus interviews in NYC") is True
    assert _guess_remote("Engineer", "This is a laboratory-based role") is False
    assert _guess_remote("Engineer", "lab-based role in South San Francisco") is False
    assert _guess_remote("Engineer", "this is a lab role") is False
    assert _guess_remote("Engineer", "this is a laboratory role") is False
    assert _guess_remote("Engineer", "work from the lab") is False
    assert _guess_remote("Engineer", "work from our laboratory") is False
    assert _guess_remote("Engineer", "this is a field role") is False
    assert _guess_remote("Engineer", "this is an office role") is False
    assert _guess_remote("Engineer", "this is an office position") is False
    assert _guess_remote("Engineer", "this is an office job") is False
    assert _guess_remote("Engineer", "work from home. this is an office role") is True
    assert _guess_remote("Engineer", "this is a headquarters role") is False
    assert _guess_remote("Engineer", "this is an HQ role") is False
    assert _guess_remote("Engineer", "must work from headquarters") is False
    assert _guess_remote("Engineer", "work from HQ") is False
    assert _guess_remote("Engineer", "work from the field") is False
    assert _guess_remote("Engineer", "work from our field office") is False
    assert _guess_remote("Engineer", "work from the field of machine learning") is True
    assert _guess_remote("Engineer", "this is a sales role") is True
    assert _guess_remote("Engineer", "field-based sales role") is False
    assert _guess_remote("Engineer", "This is a headquarters-based role") is False
    assert _guess_remote("Engineer", "HQ-based in Austin") is False
    assert _guess_remote("Engineer", "website-based application") is True
    assert _guess_remote("Engineer", "in our offices in Austin") is False
    assert _guess_remote("Engineer", "This is an in-person role") is False
    assert _guess_remote("Engineer", "must work in person") is False
    assert _guess_remote("Engineer", "in-person in NYC") is False
    assert _guess_remote("Engineer", "5 days a week in the office") is False
    assert _guess_remote("Engineer", "must come into the office") is False
    assert _guess_remote("Engineer", "come to the office") is False
    assert _guess_remote("Engineer", "come to our NYC office") is False
    assert _guess_remote("Engineer", "come into our Seattle office") is False
    assert _guess_remote("Engineer", "come to the office hours on Friday") is True
    assert _guess_remote("Engineer", "work from home. come to the office") is True
    assert _guess_remote("Engineer", "you will be based in our San Francisco office") is False
    assert _guess_remote("Engineer", "This role is based in New York") is False
    assert _guess_remote("Engineer", "This position is located in Seattle") is False
    assert _guess_remote("Engineer", "you will be based in Austin") is False
    assert _guess_remote("Engineer", "The job is based in Boston") is False
    assert _guess_remote("Engineer", "This role requires you to be in San Francisco") is False
    assert _guess_remote("Engineer", "this role requires presence in our NYC hub") is False
    assert _guess_remote("Engineer", "this role requires your presence in New York") is False
    assert _guess_remote("Engineer", "This role requires presence in the US") is True
    assert _guess_remote("Engineer", "you must be located in New York") is False
    assert _guess_remote("Engineer", "candidates must be based in Seattle") is False
    assert _guess_remote("Engineer", "This role requires you to be in the US") is True
    assert _guess_remote("Engineer", "must live in the San Francisco Bay Area") is True
    assert _guess_remote("Engineer", "must reside in California") is True
    assert _guess_remote("Engineer", "must be in Seattle 3 days a week") is False
    assert _guess_remote("Engineer", "must be in Seattle three days a week") is False
    assert _guess_remote("Engineer", "required to be in NYC 3 days a week") is False
    assert _guess_remote("Engineer", "this role is in NYC 3 days a week") is False
    assert _guess_remote("Engineer", "2-3 days a week in our NYC hub") is False
    assert _guess_remote("Engineer", "must be in the US 3 days a week") is True
    assert _guess_remote("Engineer", "3 days a week in meetings") is True
    assert _guess_remote("Engineer", "this role is hybrid") is False
    assert _guess_remote("Engineer", "this role is hybrid: 3 days in NYC") is False
    assert _guess_remote("Engineer", "the position is hybrid") is False
    assert _guess_remote("Engineer", "in-person 3 days a week") is False
    assert _guess_remote("Engineer", "in person 3 days a week") is False
    assert _guess_remote("Engineer", "3 days a week in-person") is False
    assert _guess_remote("Engineer", "3 days a week in person") is False
    assert _guess_remote("Engineer", "hybrid 3 days a week in NYC") is False
    assert _guess_remote("Engineer", "hybrid 3 days a week in the US") is True
    assert _guess_remote("Engineer", "office 3 days a week") is False
    assert _guess_remote("Engineer", "office presence 3 days a week") is False
    assert _guess_remote("Engineer", "3 days of office presence a week") is False
    assert _guess_remote("Engineer", "office attendance 3 days a week") is False
    assert _guess_remote("Engineer", "campus presence 3 days a week") is False
    assert _guess_remote("Engineer", "HQ presence 3 days a week") is False
    assert _guess_remote("Engineer", "lab presence 3 days a week") is False
    assert _guess_remote("Engineer", "3 days in Seattle per week") is False
    assert _guess_remote("Engineer", "this role requires 3 days in Seattle") is False
    assert _guess_remote("Engineer", "hybrid 3 days in NYC") is False
    assert _guess_remote("Engineer", "hybrid 3 days in the US") is True
    assert _guess_remote("Engineer", "hybrid 3 days in meetings") is True
    assert _guess_remote("Engineer", "3 days in meetings per week") is True
    assert _guess_remote("Engineer", "home office 3 days a week") is True
    assert _guess_remote("Engineer", "Microsoft Office 3 days a week") is True
    assert _guess_remote("Engineer", "this role requires 3 days in the US") is True
    assert _guess_remote("Engineer", "3 days office per week") is False
    assert _guess_remote("Engineer", "3 days a week from the office") is False
    assert _guess_remote("Engineer", "hybrid 3 days office") is False
    assert _guess_remote("Engineer", "3 days a week from home office") is True
    assert _guess_remote("Engineer", "3 days from the office each week") is False
    assert _guess_remote("Engineer", "on campus 3 days a week") is False
    assert _guess_remote("Engineer", "3 days on campus per week") is False
    assert _guess_remote("Engineer", "come into work 3 days a week") is False
    assert _guess_remote("Engineer", "come in to work 3 days a week") is False
    assert _guess_remote("Engineer", "come to work 3 days a week") is False
    assert _guess_remote("Engineer", "come in 3 days a week") is False
    assert _guess_remote("Engineer", "you'll come in 3 days a week") is False
    assert _guess_remote("Engineer", "expected to come in 3 days a week") is False
    assert _guess_remote("Engineer", "results come in 3 days a week") is True
    assert _guess_remote("Engineer", "office 3 days weekly") is False
    assert _guess_remote("Engineer", "hybrid 3 days weekly") is False
    assert _guess_remote("Engineer", "lab 3 days a week") is False
    assert _guess_remote("Engineer", "in the lab 3 days a week") is False
    assert _guess_remote("Engineer", "3 days a week from the lab") is False
    assert _guess_remote("Engineer", "campus 3 days a week") is False
    assert _guess_remote("Engineer", "3 days a week from campus") is False
    assert _guess_remote("Engineer", "3 days each week from campus") is False
    assert _guess_remote("Engineer", "3 days a week at the office") is False
    assert _guess_remote("Engineer", "3 days a week at our campus") is False
    assert _guess_remote("Engineer", "headquarters 3 days a week") is False
    assert _guess_remote("Engineer", "3 days a week from HQ") is False
    assert _guess_remote("Engineer", "report to HQ 3 days a week") is False
    assert _guess_remote("Engineer", "on-campus interviews 3 days a week") is True
    assert _guess_remote("Engineer", "you will be in Seattle 3 days a week") is True
    assert _guess_remote("Engineer", "must work from Seattle 3 days a week") is False
    assert _guess_remote("Engineer", "must work in Seattle 3 days a week") is False
    assert _guess_remote("Engineer", "must work from home 3 days a week") is True
    assert _guess_remote("Engineer", "must work in meetings 3 days a week") is True
    assert _guess_remote(
        "Engineer", "work from home. this role is hybrid"
    ) is True
    assert _guess_remote(
        "Engineer", "work from home. in-person 3 days a week"
    ) is True
    assert _guess_remote(
        "Engineer", "work from home. office 3 days a week"
    ) is True
    assert _guess_remote(
        "Engineer", "work from home. hybrid 3 days in NYC"
    ) is True
    assert _guess_remote(
        "Engineer", "work from home. 3 days office per week"
    ) is True
    assert _guess_remote("Engineer", "in-person interviews 3 days a week") is True
    assert _guess_remote(
        "Engineer", "work from home. must be in Seattle 3 days a week"
    ) is True
    assert _guess_remote("Engineer", "relocation to Seattle required") is False
    assert _guess_remote("Engineer", "Relocation to New York City is required") is False
    assert _guess_remote("Engineer", "this role requires relocation to NYC") is False
    assert _guess_remote("Engineer", "this position requires relocation") is False
    assert _guess_remote("Engineer", "must relocate to Seattle") is False
    assert _guess_remote("Engineer", "you are required to relocate to Austin") is False
    assert _guess_remote("Engineer", "you are not required to relocate to Austin") is True
    assert _guess_remote("Engineer", "relocation is required") is False
    assert _guess_remote("Engineer", "this role requires relocation to the US") is True
    assert _guess_remote("Engineer", "must relocate to the US") is True
    assert _guess_remote("Engineer", "no relocation is required") is True
    assert _guess_remote("Engineer", "relocation is not required") is True
    assert _guess_remote("Engineer", "this role requires relocation assistance") is True
    assert _guess_remote("Engineer", "Salary $180,000 relocation to Seattle") is True
    assert _guess_remote(
        "Engineer", "work from home. this role requires relocation to NYC"
    ) is True
    assert _guess_remote("Engineer", "This role is based in the US") is True
    assert _guess_remote("Engineer", "We're based in New York. Great team.") is True
    assert _guess_remote("Engineer", "Microsoft Office 365 and Slack") is True
    assert _guess_remote("Engineer", "work from home") is True
    assert _guess_remote("Engineer", "work from home. office-first role") is True
    assert _guess_remote("Engineer", "work from home. This is a site-based role") is True
    assert _guess_remote("Engineer", "work from home. This is an on-campus role") is True
    assert _guess_remote("Engineer", "work from home. this is a lab role") is True
    assert _guess_remote("Engineer", "work from home. work from the lab") is True
    assert _guess_remote("Engineer", "work from home. this is a field role") is True
    assert _guess_remote("Engineer", "work from home. work from HQ") is True
    assert _guess_remote("Engineer", "work from home. work from our New York office") is True
    assert _guess_remote("Engineer", "work from home. work out of the Seattle office") is True
    assert _guess_remote(
        "Engineer", "work from home. This role requires you to be in San Francisco"
    ) is True
    assert _guess_remote(
        "Engineer", "work from home. this role requires presence in our NYC hub"
    ) is True
    assert _guess_remote(
        "Engineer", "work from home. must commute to San Francisco"
    ) is True
    office = Opportunity(title="Engineer", url="https://jobs.example/off")
    assert _apply_listing(
        office, "<p>Work from our office in NYC. Salary $180,000</p>"
    ) is True
    assert office.remote is False
    assert office.pay_high == 180_000
    assert office.score() == 0.7 * (180_000 / (40 * 50))
    city_office = Opportunity(title="Engineer", url="https://jobs.example/city-off")
    assert _apply_listing(
        city_office, "<p>Work from our New York office. Salary $180,000</p>"
    ) is True
    assert city_office.remote is False
    assert city_office.pay_high == 180_000
    assert city_office.score() == 0.7 * (180_000 / (40 * 50))
    out_of = Opportunity(title="Engineer", url="https://jobs.example/out-of")
    assert _apply_listing(
        out_of, "<p>Work out of the Seattle office. Salary $180,000</p>"
    ) is True
    assert out_of.remote is False
    assert out_of.pay_high == 180_000
    assert out_of.score() == 0.7 * (180_000 / (40 * 50))
    come = Opportunity(title="Engineer", url="https://jobs.example/come")
    assert _apply_listing(
        come, "<p>Come to the office. Salary $180,000</p>"
    ) is True
    assert come.remote is False
    assert come.pay_high == 180_000
    assert come.score() == 0.7 * (180_000 / (40 * 50))
    ofirst = Opportunity(title="Engineer", url="https://jobs.example/ofirst")
    assert _apply_listing(
        ofirst, "<p>This is an office-first role. Salary $180,000</p>"
    ) is True
    assert ofirst.remote is False
    assert ofirst.pay_high == 180_000
    assert ofirst.score() == 0.7 * (180_000 / (40 * 50))
    days = Opportunity(title="Engineer", url="https://jobs.example/days")
    assert _apply_listing(
        days, "<p>5 days a week in the office. Salary $180,000</p>"
    ) is True
    assert days.remote is False
    assert days.score() == 0.7 * (180_000 / (40 * 50))
    based = Opportunity(title="Engineer", url="https://jobs.example/based")
    assert _apply_listing(
        based, "<p>This role is based in New York. Salary $180,000</p>"
    ) is True
    assert based.remote is False
    assert based.score() == 0.7 * (180_000 / (40 * 50))
    require = Opportunity(title="Engineer", url="https://jobs.example/require")
    assert _apply_listing(
        require,
        "<p>This role requires you to be in San Francisco. Salary $180,000</p>",
    ) is True
    assert require.remote is False
    assert require.pay_high == 180_000
    assert require.score() == 0.7 * (180_000 / (40 * 50))
    presence = Opportunity(title="Engineer", url="https://jobs.example/presence")
    assert _apply_listing(
        presence,
        "<p>This role requires presence in our NYC hub. Salary $180,000</p>",
    ) is True
    assert presence.remote is False
    assert presence.pay_high == 180_000
    assert presence.score() == 0.7 * (180_000 / (40 * 50))
    located = Opportunity(title="Engineer", url="https://jobs.example/located")
    assert _apply_listing(
        located, "<p>You must be located in New York. Salary $180,000</p>"
    ) is True
    assert located.remote is False
    assert located.score() == 0.7 * (180_000 / (40 * 50))
    site = Opportunity(title="Engineer", url="https://jobs.example/site")
    assert _apply_listing(
        site, "<p>This is a site-based role. Salary $180,000</p>"
    ) is True
    assert site.remote is False
    assert site.pay_high == 180_000
    assert site.score() == 0.7 * (180_000 / (40 * 50))
    campus = Opportunity(title="Engineer", url="https://jobs.example/campus")
    assert _apply_listing(
        campus, "<p>This is a campus-based position. Salary $180,000</p>"
    ) is True
    assert campus.remote is False
    assert campus.score() == 0.7 * (180_000 / (40 * 50))
    oncampus = Opportunity(title="Engineer", url="https://jobs.example/oncampus")
    assert _apply_listing(
        oncampus, "<p>This is an on-campus role. Salary $180,000</p>"
    ) is True
    assert oncampus.remote is False
    assert oncampus.pay_high == 180_000
    assert oncampus.score() == 0.7 * (180_000 / (40 * 50))
    lab = Opportunity(title="Engineer", url="https://jobs.example/lab")
    assert _apply_listing(
        lab, "<p>This is a lab role. Salary $180,000</p>"
    ) is True
    assert lab.remote is False
    assert lab.pay_high == 180_000
    assert lab.score() == 0.7 * (180_000 / (40 * 50))
    fromlab = Opportunity(title="Engineer", url="https://jobs.example/fromlab")
    assert _apply_listing(
        fromlab, "<p>Work from the lab. Salary $180,000</p>"
    ) is True
    assert fromlab.remote is False
    assert fromlab.score() == 0.7 * (180_000 / (40 * 50))
    field = Opportunity(title="Engineer", url="https://jobs.example/field")
    assert _apply_listing(
        field, "<p>This is a field role. Salary $180,000</p>"
    ) is True
    assert field.remote is False
    assert field.pay_high == 180_000
    assert field.score() == 0.7 * (180_000 / (40 * 50))
    office_role = Opportunity(title="Engineer", url="https://jobs.example/office-role")
    assert _apply_listing(
        office_role, "<p>This is an office role. Salary $180,000</p>"
    ) is True
    assert office_role.remote is False
    assert office_role.pay_high == 180_000
    assert office_role.score() == 0.7 * (180_000 / (40 * 50))
    hq = Opportunity(title="Engineer", url="https://jobs.example/hq")
    assert _apply_listing(
        hq, "<p>Must work from headquarters. Salary $180,000</p>"
    ) is True
    assert hq.remote is False
    assert hq.score() == 0.7 * (180_000 / (40 * 50))
    report = Opportunity(title="Engineer", url="https://jobs.example/report")
    assert _apply_listing(
        report, "<p>Must report to our NYC office. Salary $180,000</p>"
    ) is True
    assert report.remote is False
    assert report.score() == 0.7 * (180_000 / (40 * 50))
    commute = Opportunity(title="Engineer", url="https://jobs.example/commute")
    assert _apply_listing(
        commute, "<p>Must commute to San Francisco. Salary $180,000</p>"
    ) is True
    assert commute.remote is False
    assert commute.pay_high == 180_000
    assert commute.score() == 0.7 * (180_000 / (40 * 50))
    reloc = Opportunity(title="Engineer", url="https://jobs.example/reloc")
    assert _apply_listing(
        reloc, "<p>Relocation to Seattle required. Salary $180,000</p>"
    ) is True
    assert reloc.remote is False
    assert reloc.pay_high == 180_000
    assert reloc.score() == 0.7 * (180_000 / (40 * 50))
    requires_reloc = Opportunity(title="Engineer", url="https://jobs.example/req-reloc")
    assert _apply_listing(
        requires_reloc,
        "<p>This role requires relocation to NYC. Salary $180,000</p>",
    ) is True
    assert requires_reloc.remote is False
    assert requires_reloc.score() == 0.7 * (180_000 / (40 * 50))
    days_city = Opportunity(title="Engineer", url="https://jobs.example/days-city")
    assert _apply_listing(
        days_city, "<p>Must be in Seattle 3 days a week. Salary $180,000</p>"
    ) is True
    assert days_city.remote is False
    assert days_city.pay_high == 180_000
    assert days_city.score() == 0.7 * (180_000 / (40 * 50))
    hub_days = Opportunity(title="Engineer", url="https://jobs.example/hub-days")
    assert _apply_listing(
        hub_days, "<p>2-3 days a week in our NYC hub. Salary $180,000</p>"
    ) is True
    assert hub_days.remote is False
    assert hub_days.score() == 0.7 * (180_000 / (40 * 50))
    role_hybrid = Opportunity(title="Engineer", url="https://jobs.example/role-hyb")
    assert _apply_listing(
        role_hybrid, "<p>This role is hybrid: 3 days in NYC. Salary $180,000</p>"
    ) is True
    assert role_hybrid.remote is False
    assert role_hybrid.score() == 0.7 * (180_000 / (40 * 50))
    inperson_days = Opportunity(title="Engineer", url="https://jobs.example/ip-days")
    assert _apply_listing(
        inperson_days, "<p>In-person 3 days a week. Salary $180,000</p>"
    ) is True
    assert inperson_days.remote is False
    assert inperson_days.score() == 0.7 * (180_000 / (40 * 50))
    days_inperson = Opportunity(title="Engineer", url="https://jobs.example/days-ip")
    assert _apply_listing(
        days_inperson, "<p>3 days a week in-person. Salary $180,000</p>"
    ) is True
    assert days_inperson.remote is False
    assert days_inperson.score() == 0.7 * (180_000 / (40 * 50))
    hybrid_city = Opportunity(title="Engineer", url="https://jobs.example/hyb-city")
    assert _apply_listing(
        hybrid_city, "<p>Hybrid 3 days a week in NYC. Salary $180,000</p>"
    ) is True
    assert hybrid_city.remote is False
    assert hybrid_city.score() == 0.7 * (180_000 / (40 * 50))
    work_from_city = Opportunity(title="Engineer", url="https://jobs.example/wfc-days")
    assert _apply_listing(
        work_from_city, "<p>Must work from Seattle 3 days a week. Salary $180,000</p>"
    ) is True
    assert work_from_city.remote is False
    assert work_from_city.score() == 0.7 * (180_000 / (40 * 50))
    office_days = Opportunity(title="Engineer", url="https://jobs.example/off-days")
    assert _apply_listing(
        office_days, "<p>Office 3 days a week. Salary $180,000</p>"
    ) is True
    assert office_days.remote is False
    assert office_days.score() == 0.7 * (180_000 / (40 * 50))
    presence = Opportunity(title="Engineer", url="https://jobs.example/off-pres")
    assert _apply_listing(
        presence, "<p>Office presence 3 days a week. Salary $180,000</p>"
    ) is True
    assert presence.remote is False
    assert presence.score() == 0.7 * (180_000 / (40 * 50))
    city_week = Opportunity(title="Engineer", url="https://jobs.example/city-week")
    assert _apply_listing(
        city_week, "<p>3 days in Seattle per week. Salary $180,000</p>"
    ) is True
    assert city_week.remote is False
    assert city_week.score() == 0.7 * (180_000 / (40 * 50))
    hybrid_in = Opportunity(title="Engineer", url="https://jobs.example/hyb-in")
    assert _apply_listing(
        hybrid_in, "<p>Hybrid 3 days in NYC. Salary $180,000</p>"
    ) is True
    assert hybrid_in.remote is False
    assert hybrid_in.score() == 0.7 * (180_000 / (40 * 50))
    req_days = Opportunity(title="Engineer", url="https://jobs.example/req-days")
    assert _apply_listing(
        req_days, "<p>This role requires 3 days in Seattle. Salary $180,000</p>"
    ) is True
    assert req_days.remote is False
    assert req_days.score() == 0.7 * (180_000 / (40 * 50))
    office_rev = Opportunity(title="Engineer", url="https://jobs.example/off-rev")
    assert _apply_listing(
        office_rev, "<p>3 days office per week. Salary $180,000</p>"
    ) is True
    assert office_rev.remote is False
    assert office_rev.score() == 0.7 * (180_000 / (40 * 50))
    from_off = Opportunity(title="Engineer", url="https://jobs.example/from-off")
    assert _apply_listing(
        from_off, "<p>3 days a week from the office. Salary $180,000</p>"
    ) is True
    assert from_off.remote is False
    assert from_off.score() == 0.7 * (180_000 / (40 * 50))
    each_week = Opportunity(title="Engineer", url="https://jobs.example/each-wk")
    assert _apply_listing(
        each_week, "<p>3 days from the office each week. Salary $180,000</p>"
    ) is True
    assert each_week.remote is False
    assert each_week.score() == 0.7 * (180_000 / (40 * 50))
    come_work = Opportunity(title="Engineer", url="https://jobs.example/come-work")
    assert _apply_listing(
        come_work, "<p>Come into work 3 days a week. Salary $180,000</p>"
    ) is True
    assert come_work.remote is False
    assert come_work.score() == 0.7 * (180_000 / (40 * 50))
    come_to = Opportunity(title="Engineer", url="https://jobs.example/come-to")
    assert _apply_listing(
        come_to, "<p>Come to work 3 days a week. Salary $180,000</p>"
    ) is True
    assert come_to.remote is False
    at_off = Opportunity(title="Engineer", url="https://jobs.example/at-off")
    assert _apply_listing(
        at_off, "<p>3 days a week at the office. Salary $180,000</p>"
    ) is True
    assert at_off.remote is False
    come_in = Opportunity(title="Engineer", url="https://jobs.example/come-in")
    assert _apply_listing(
        come_in, "<p>Come in 3 days a week. Salary $180,000</p>"
    ) is True
    assert come_in.remote is False
    weekly_off = Opportunity(title="Engineer", url="https://jobs.example/wk-off")
    assert _apply_listing(
        weekly_off, "<p>Office 3 days weekly. Salary $180,000</p>"
    ) is True
    assert weekly_off.remote is False
    lab_days = Opportunity(title="Engineer", url="https://jobs.example/lab-days")
    assert _apply_listing(
        lab_days, "<p>Lab 3 days a week. Salary $180,000</p>"
    ) is True
    assert lab_days.remote is False
    assert _guess_remote("Engineer", "fully distributed team") is True  # default
    assert _guess_remote("Engineer", "This role can be hybrid, or fully remote/virtually.") is True
    assert _guess_remote("Engineer", "Build hybrid retrieval and hybrid models.") is True
    assert _guess_remote("Engineer", "Our structured hybrid approach is centered around our offices") is False
    assert _guess_remote(
        "Engineer",
        "The work style of each role, Hybrid, Remote, or In-Person is indicated in the job description.",
    ) is True
    assert _guess_remote(
        "Engineer",
        "Apply now. Similar Jobs Square Account Executive Remote or Hybrid Everett, WA",
    ) is True


# --- DuckDuckGo HTML parsing -------------------------------------------


DDG_HTML = """
<div class="links_main">
  <a class="result__a" href="https://example.com/job1">Senior ML Engineer</a>
  <a class="result__snippet" href="https://example.com/job1">Remote role, great pay</a>
</div>
<div class="links_main">
  <a class="result__a" href="//example.org/job2">Data Scientist</a>
</div>
<div class="links_main">
  <a class="result__a" href="https://duckduckgo.com/y.js?ad=1">Sponsored</a>
</div>
"""


def test_parse_ddg_extracts_title_url_and_snippet():
    results = _parse_ddg_html(DDG_HTML)

    # ad link (y.js) filtered out -> 2 real results
    assert len(results) == 2

    first = results[0]
    assert first["url"] == "https://example.com/job1"
    assert first["title"] == "Senior ML Engineer"
    assert first["description"] == "Remote role, great pay"
    assert first["source"] == "duckduckgo"


def test_parse_ddg_normalizes_protocol_relative_url():
    results = _parse_ddg_html(DDG_HTML)
    assert results[1]["url"] == "https://example.org/job2"


def test_parse_ddg_empty_input():
    assert _parse_ddg_html("") == []


DDG_LIVE_SHAPE = """
<a class="result__a" href="https://www.indeed.com/q-ml-engineer-remote-$150,000-jobs.html">Flexible Ml Engineer Remote $150,000 Jobs - Indeed</a>
<div class="result__extras">
  <a rel="nofollow" href="https://www.indeed.com/q-ml-engineer-remote-$150,000-jobs.html">
    <img class="result__icon__img" width="16" height="16" alt="" src="//external-content.duckduckgo.com/ip3/www.indeed.com.ico" />
  </a>
  <a class="result__url" href="https://www.indeed.com/q-ml-engineer-remote-$150,000-jobs.html">
    www.indeed.com/q-ml-engineer-remote-$150,000-jobs.html
  </a>
</div>
<a class="result__snippet" href="https://www.indeed.com/q-ml-engineer-remote-$150,000-jobs.html">Browse 568 <b>Ml</b> <b>Engineer</b> <b>Remote</b> $150,000 job openings. Discover flexible, work-from-home opportunities.</a>
<a class="result__a" href="https://jobs.example/onsite">Office Role</a>
<a class="result__snippet" href="https://jobs.example/onsite">Must be <b>hybrid</b>, $80/hr, 20 hrs/week</a>
"""


def test_parse_ddg_strips_bold_and_does_not_need_a_tiny_window():
    results = _parse_ddg_html(DDG_LIVE_SHAPE)
    assert len(results) == 2
    assert results[0]["description"] == (
        "Browse 568 Ml Engineer Remote $150,000 job openings. "
        "Discover flexible, work-from-home opportunities."
    )
    assert results[1]["description"] == "Must be hybrid, $80/hr, 20 hrs/week"


def test_search_ddg_retries_202_then_parses(monkeypatch):
    import httpx

    hits: list[int] = []

    class FakeResp:
        def __init__(self, status_code: int, text: str):
            self.status_code = status_code
            self.text = text

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return None

        async def post(self, _url, **_kwargs):
            hits.append(1)
            if len(hits) == 1:
                return FakeResp(202, "<html>challenge</html>")
            return FakeResp(200, DDG_HTML)

        async def get(self, _url, **_kwargs):
            return FakeResp(202, "<html>challenge</html>")

    monkeypatch.setattr(httpx, "AsyncClient", lambda **_k: FakeClient())

    async def no_sleep(_delay):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    rows = asyncio.run(Engine()._search_ddg("ml"))
    assert len(hits) == 2
    assert [r["url"] for r in rows] == [
        "https://example.com/job1",
        "https://example.org/job2",
    ]


def test_search_ddg_gives_up_after_202s(monkeypatch):
    import httpx

    hits: list[int] = []

    class FakeResp:
        status_code = 202
        text = "<html>challenge</html>"

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return None

        async def post(self, _url, **_kwargs):
            hits.append(1)
            return FakeResp()

        async def get(self, _url, **_kwargs):
            return FakeResp()

    monkeypatch.setattr(httpx, "AsyncClient", lambda **_k: FakeClient())

    async def no_sleep(_delay):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    assert asyncio.run(Engine()._search_ddg("ml")) == []
    assert len(hits) == 4


def test_search_ddg_retries_200_without_results(monkeypatch):
    import httpx

    hits: list[int] = []

    class FakeResp:
        def __init__(self, status_code: int, text: str):
            self.status_code = status_code
            self.text = text

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return None

        async def post(self, _url, **_kwargs):
            hits.append(1)
            if len(hits) == 1:
                return FakeResp(200, "<html>challenge</html>")
            return FakeResp(200, DDG_HTML)

        async def get(self, _url, **_kwargs):
            return FakeResp(202, "<html>challenge</html>")

    monkeypatch.setattr(httpx, "AsyncClient", lambda **_k: FakeClient())

    async def no_sleep(_delay):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    rows = asyncio.run(Engine()._search_ddg("ml"))
    assert len(hits) == 2
    assert rows[0]["url"] == "https://example.com/job1"


DDG_LITE_HTML = """
<table border="0">
  <tr>
    <td>1.&nbsp;</td>
    <td>
      <a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fjobs.ashbyhq.com%2Fquilter%2F2b0f95cb-7c8b-4b62-8bcb-b9993344f2f1&amp;rut=abc" class='result-link'>Senior ML Engineer @ Quilter</a>
    </td>
  </tr>
  <tr>
    <td>&nbsp;</td>
    <td class='result-snippet'><b>Senior</b> <b>ML</b> Engineer. Remote. $180K – $200K.</td>
  </tr>
  <tr>
    <td>2.&nbsp;</td>
    <td>
      <a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fjobs.lever.co%2Fswordhealth%2F50945411-2f43-421a-8bb8-86aa1de6d890&amp;rut=def" class='result-link'>Sword Health - Senior ML</a>
    </td>
  </tr>
</table>
"""


def test_parse_ddg_lite_unwraps_uddg_and_snippets():
    results = _parse_ddg_html(DDG_LITE_HTML)
    assert [r["url"] for r in results] == [
        "https://jobs.ashbyhq.com/quilter/2b0f95cb-7c8b-4b62-8bcb-b9993344f2f1",
        "https://jobs.lever.co/swordhealth/50945411-2f43-421a-8bb8-86aa1de6d890",
    ]
    assert results[0]["title"] == "Senior ML Engineer @ Quilter"
    assert results[0]["description"] == "Senior ML Engineer. Remote. $180K – $200K."
    assert results[1]["description"] == ""


def test_search_ddg_falls_back_to_lite_when_html_202s(monkeypatch):
    import httpx

    posts: list[int] = []
    gets: list[str] = []

    class FakeResp:
        def __init__(self, status_code: int, text: str):
            self.status_code = status_code
            self.text = text

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return None

        async def post(self, _url, **_kwargs):
            posts.append(1)
            return FakeResp(202, "<html>challenge</html>")

        async def get(self, url, **kwargs):
            gets.append(url)
            assert kwargs.get("params", {}).get("q") == "ml"
            return FakeResp(200, DDG_LITE_HTML)

    monkeypatch.setattr(httpx, "AsyncClient", lambda **_k: FakeClient())
    rows = asyncio.run(Engine()._search_ddg("ml"))
    assert posts == [1]
    assert gets == ["https://lite.duckduckgo.com/lite/"]
    assert [r["url"] for r in rows] == [
        "https://jobs.ashbyhq.com/quilter/2b0f95cb-7c8b-4b62-8bcb-b9993344f2f1",
        "https://jobs.lever.co/swordhealth/50945411-2f43-421a-8bb8-86aa1de6d890",
    ]


def test_heuristic_uses_ddg_snippet_pay_hours_and_remote():
    results = _parse_ddg_html(DDG_LIVE_SHAPE)
    office = _heuristic_opportunity(results[1])
    assert office.pay_high == 80_000
    assert office.hours_per_week == 20
    assert office.remote is False
    assert office.score() == 56.0  # 80k / (20*50) * 0.7 office


# --- search aggregation -------------------------------------------------


def test_search_all_dedupes_by_url():
    engine = Engine()

    async def fake_brave(_query: str):
        return [
            {"url": "https://a.com/x", "title": "A"},
            {"url": "https://b.com/y", "title": "B"},
            {"url": "https://a.com/x", "title": "A duplicate"},
        ]

    async def fake_perplexity(_query: str):
        return []

    engine._search_brave = fake_brave
    engine._search_perplexity = fake_perplexity

    results = asyncio.run(engine._search_all("anything"))
    urls = [r["url"] for r in results]

    assert urls == ["https://a.com/x", "https://b.com/y"]


def test_search_all_dedupes_normalized_urls():
    engine = Engine()

    async def fake_brave(_query: str):
        return [
            {"url": "https://a.com/x/", "title": "A slash"},
            {"url": "HTTPS://A.COM/X", "title": "A case"},
            {"url": "https://b.com/y", "title": "B"},
        ]

    async def fake_perplexity(_query: str):
        return []

    engine._search_brave = fake_brave
    engine._search_perplexity = fake_perplexity

    results = asyncio.run(engine._search_all("anything"))
    assert [r["url"] for r in results] == ["https://a.com/x/", "https://b.com/y"]


def test_search_all_dedupes_lever_apply_to_job_url():
    engine = Engine()

    async def fake_brave(_query: str):
        return [
            {
                "title": "Apply",
                "url": "https://jobs.lever.co/provectus/0bf1decc-002c-4b0a-b97b-6407d2930fff/apply",
            },
            {
                "title": "Job",
                "url": "https://jobs.lever.co/provectus/0bf1decc-002c-4b0a-b97b-6407d2930fff",
            },
        ]

    async def fake_perplexity(_query: str):
        return []

    engine._search_brave = fake_brave
    engine._search_perplexity = fake_perplexity
    results = asyncio.run(engine._search_all("ml"))
    assert [r["url"] for r in results] == [
        "https://jobs.lever.co/provectus/0bf1decc-002c-4b0a-b97b-6407d2930fff/apply"
    ]


def test_search_all_drops_failed_sources():
    engine = Engine()

    async def fake_brave(_query: str):
        raise RuntimeError("source down")

    async def fake_perplexity(_query: str):
        return []

    engine._search_brave = fake_brave
    engine._search_perplexity = fake_perplexity

    # gather(return_exceptions=True) -> exceptions ignored, no crash
    assert asyncio.run(engine._search_all("anything")) == []


def _fake_client(content: str, captured: dict | None = None):
    async def create(**kwargs):
        if captured is not None:
            captured.update(kwargs)
        message = types.SimpleNamespace(content=content)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    return types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))
    )


def _fake_client_raises(exc: Exception):
    async def create(**kwargs):
        raise exc

    return types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))
    )


def test_heuristic_opportunity_requires_url():
    assert _heuristic_opportunity({"title": "No url", "pay": 100_000}) is None


def test_heuristic_opportunity_prefers_raw_then_guesses():
    raw = {
        "title": "Staff Engineer",
        "company": "Acme",
        "url": "https://example.com/job",
        "description": "onsite hybrid",
        "pay": 200_000,
        "hours": 25,
        "remote": False,
        "source": "brave",
    }
    opp = _heuristic_opportunity(raw)
    assert isinstance(opp, Opportunity)
    assert opp.pay_high == 200_000
    assert opp.hours_per_week == 25
    assert opp.remote is False
    assert opp.efficiency == opp.refined_rate == 160.0

    guessed = _heuristic_opportunity(
        {
            "title": "Senior ML Engineer $180k",
            "url": "https://example.com/senior",
            "description": "must be onsite, 20 hrs/week",
            "source": "ddg",
        }
    )
    assert guessed.pay_high == 180_000
    assert guessed.hours_per_week == 20
    assert guessed.remote is False
    assert guessed.rate_is_imputed is False

    thin = _heuristic_opportunity(
        {
            "title": "Senior ML Engineer",
            "url": "https://example.com/thin",
            "description": "must be onsite",
            "source": "ddg",
        }
    )
    assert thin.pay_high is None
    assert thin.hours_per_week is None
    assert thin.score() == 0
    assert thin.remote is False


def test_index_pages_are_not_opportunities():
    assert (
        _heuristic_opportunity(
            {
                "title": "Flexible Ml Engineer Remote $150,000 Jobs - Indeed",
                "url": "https://www.indeed.com/q-ml-engineer-remote-$150,000-jobs.html",
                "description": "Browse 568 Ml Engineer Remote $150,000 job openings.",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Senior Machine Learning Engineer - Acme - Indeed",
                "url": "https://www.indeed.com/viewjob?jk=abc123def456",
                "description": "$180,000 - $220,000 a year",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "ML Engineer - Acme",
                "url": "https://ca.indeed.com/viewjob?jk=xyz789",
                "description": "$90,000 a year",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Senior Machine Learning Engineer - Acme - Monster",
                "url": "https://www.monster.com/job-openings/senior-machine-learning-engineer-acme-san-francisco-ca",
                "description": "$180,000 - $220,000 a year",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Senior ML Engineer | Acme | Dice.com",
                "url": "https://www.dice.com/job-detail/abc123",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Senior Machine Learning Engineer - Acme",
                "url": "https://www.simplyhired.com/job/abc123",
                "description": "$180,000 - $220,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Senior ML Engineer - Acme - Jooble",
                "url": "https://jooble.org/j/123456789",
                "description": "$180,000 a year",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "ML Engineer $180,000 - Adzuna",
                "url": "https://www.adzuna.com/details/123456",
                "description": "$180,000 - $220,000 a year",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Senior ML Engineer - Acme | Talent.com",
                "url": "https://www.talent.com/view?id=abc123",
                "description": "$180,000 a year",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Senior ML Engineer - Acme - CareerBuilder",
                "url": "https://www.careerbuilder.com/job/abc123",
                "description": "$180,000 - $220,000 a year",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Remote Machine Learning Engineer Jobs ($104K-$225K)",
                "url": "https://www.remoterocketship.com/jobs/machine-learning-engineer/",
                "description": "Search 546 remote jobs.",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "RemoteFront | 100,000+ Remote Jobs from 20,000+ Vetted Companies",
                "url": "https://www.remotefront.com/remote-ml-engineer-jobs",
                "description": "median $190k (most $150k-$215k)",
            }
        )
        is None
    )
    kept = _heuristic_opportunity(
        {
            "title": "Senior AI/ML Engineer",
            "url": "https://www.gravityer.com/jobs/ctg-senior-ai-ml-engineer",
            "description": "Remote (US Only) | $150K-$200K",
        }
    )
    assert kept is not None
    assert kept.pay_high == 200_000
    lever = _heuristic_opportunity(
        {
            "title": "Lyra Health - Senior ML Engineer (ML/AI) - jobs.lever.co",
            "url": "https://jobs.lever.co/lyrahealth/d33ddfed-8c69-4e29-966b-0e190190cd6a",
            "description": "Remote role.",
        }
    )
    assert lever is not None
    assert lever.title == "Lyra Health - Senior ML Engineer (ML/AI)"
    gh_app = _heuristic_opportunity(
        {
            "title": "Job Application for Senior, ML Engineer - VLM at Torc Robotics",
            "url": "https://job-boards.greenhouse.io/torcrobotics/jobs/8572505002",
            "description": "",
        }
    )
    assert gh_app is not None
    assert gh_app.title == "Senior, ML Engineer - VLM at Torc Robotics"
    assert gh_app.company == "Torc Robotics"
    workable = _heuristic_opportunity(
        {
            "title": "Senior ML Engineer | Canopy | Jobs By Workable",
            "url": "https://jobs.workable.com/view/7mMjfHgS93LyPeHLK2XeMV/remote-senior-machine-learning-engineer-in-detroit-at-canopy",
            "description": "Remote role.",
        }
    )
    assert workable is not None
    assert workable.company == "Canopy"
    assert workable.title == "Senior ML Engineer | Canopy"
    assert (
        _heuristic_opportunity(
            {
                "title": "Intuition Machines, Inc. - Current Openings",
                "url": "https://apply.workable.com/imachines",
                "description": "",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "A2Z Sync - Current Openings",
                "url": "https://apply.workable.com/a2z-sync/",
                "description": "",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Open Positions | Stripe",
                "url": "https://stripe.com/jobs/open-positions",
                "description": "$180,000 - $270,000. Remote.",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Software Engineer",
                "url": "https://acme.com/careers/open-roles",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Open Position: Software Engineer",
                "url": "https://jobs.example.com/job/open-position-software-engineer",
                "description": "$180,000",
            }
        )
        is not None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Career Opportunities | Acme",
                "url": "https://acme.com/career-opportunities",
                "description": "$180,000 - $270,000. Remote.",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Job Openings | Stripe",
                "url": "https://acme.com/teams/job-openings",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Careers at Acme",
                "url": "https://acme.com/about/careers-team",
                "description": "$200,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Software Engineer",
                "url": "https://acme.com/job-openings/senior-engineer",
                "description": "$180,000",
            }
        )
        is not None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Job Opening: Software Engineer",
                "url": "https://jobs.example.com/job/job-opening-software-engineer",
                "description": "$180,000",
            }
        )
        is not None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Join Our Team | Acme",
                "url": "https://acme.com/join-our-team",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Work With Us | Acme",
                "url": "https://acme.com/work-with-us",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "We're Hiring | Acme",
                "url": "https://acme.com/were-hiring",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Opportunities | Acme",
                "url": "https://acme.com/opportunities",
                "description": "$200,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Join Our Team as a Software Engineer",
                "url": "https://jobs.example.com/job/join-our-team-software-engineer",
                "description": "$180,000",
            }
        )
        is not None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Software Engineer",
                "url": "https://acme.com/join-our-team/senior-engineer",
                "description": "$180,000",
            }
        )
        is not None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Job Vacancies | Acme",
                "url": "https://acme.com/vacancies",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Available Positions",
                "url": "https://acme.com/available-positions",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Hiring | Acme",
                "url": "https://acme.com/hiring",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Explore Careers",
                "url": "https://acme.com/explore-careers",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Hiring Manager",
                "url": "https://jobs.example.com/job/hiring-manager",
                "description": "$180,000",
            }
        )
        is not None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Life at Acme",
                "url": "https://acme.com/life-at-acme",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Meet the Team | Acme",
                "url": "https://acme.com/meet-the-team",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Our People",
                "url": "https://acme.com/our-people",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Our Team Lead",
                "url": "https://jobs.example.com/job/our-team-lead",
                "description": "$180,000",
            }
        )
        is not None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Team | Acme",
                "url": "https://acme.com/team",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Why Acme",
                "url": "https://acme.com/about",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Internships | Acme",
                "url": "https://acme.com/internships",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "University Recruiting",
                "url": "https://acme.com/university-recruiting",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Campus Recruiting | Acme",
                "url": "https://acme.com/campus-recruiting",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Early Careers | Acme",
                "url": "https://acme.com/early-careers",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Student Programs",
                "url": "https://acme.com/student-programs",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Graduate Programs | Acme",
                "url": "https://acme.com/graduate-programs",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "University Programs",
                "url": "https://acme.com/university-programs",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Job Search | Acme",
                "url": "https://acme.com/job-search",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Careers | Acme",
                "url": "https://acme.com/about/careers-overview",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Benefits | Acme",
                "url": "https://acme.com/benefits",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Our Benefits",
                "url": "https://acme.com/our-benefits",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Benefits Analyst",
                "url": "https://jobs.example.com/job/benefits-analyst",
                "description": "$180,000",
            }
        )
        is not None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Culture | Acme",
                "url": "https://acme.com/culture",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Our Culture",
                "url": "https://acme.com/our-culture",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Leadership | Acme",
                "url": "https://acme.com/leadership",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Culture Engineer",
                "url": "https://jobs.example.com/job/culture-engineer",
                "description": "$180,000",
            }
        )
        is not None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Leadership Development Program",
                "url": "https://jobs.example.com/job/ldp",
                "description": "$180,000",
            }
        )
        is not None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "About Us | Acme",
                "url": "https://acme.com/about-us",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Our Values | Acme",
                "url": "https://acme.com/our-values",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Locations | Acme",
                "url": "https://acme.com/locations",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Values Engineer",
                "url": "https://jobs.example.com/job/values-engineer",
                "description": "$180,000",
            }
        )
        is not None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "About this Role",
                "url": "https://jobs.example.com/job/about-this-role",
                "description": "$180,000",
            }
        )
        is not None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Diversity | Acme",
                "url": "https://acme.com/diversity",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Inclusion | Acme",
                "url": "https://acme.com/inclusion",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "DEI | Acme",
                "url": "https://acme.com/dei",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Diversity Engineer",
                "url": "https://jobs.example.com/job/diversity-engineer",
                "description": "$180,000",
            }
        )
        is not None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Our Story | Acme",
                "url": "https://acme.com/our-story",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "FAQs | Acme",
                "url": "https://acme.com/faqs",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Story Engineer",
                "url": "https://jobs.example.com/job/story-engineer",
                "description": "$180,000",
            }
        )
        is not None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "News | Acme",
                "url": "https://acme.com/news",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Press | Acme",
                "url": "https://acme.com/press",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "News Engineer",
                "url": "https://jobs.example.com/job/news-engineer",
                "description": "$180,000",
            }
        )
        is not None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Press Engineer",
                "url": "https://jobs.example.com/job/press-engineer",
                "description": "$180,000",
            }
        )
        is not None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Newsroom | Acme",
                "url": "https://acme.com/newsroom",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Investors | Acme",
                "url": "https://acme.com/investors",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Investors Engineer",
                "url": "https://jobs.example.com/job/investors-engineer",
                "description": "$180,000",
            }
        )
        is not None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Sustainability | Acme",
                "url": "https://acme.com/sustainability",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Sustainability Engineer",
                "url": "https://jobs.example.com/job/sustainability-engineer",
                "description": "$180,000",
            }
        )
        is not None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "ESG | Acme",
                "url": "https://acme.com/esg",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Impact | Acme",
                "url": "https://acme.com/impact",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Community | Acme",
                "url": "https://acme.com/community",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Impact Engineer",
                "url": "https://jobs.example.com/job/impact-engineer",
                "description": "$180,000",
            }
        )
        is not None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Community Engineer",
                "url": "https://jobs.example.com/job/community-engineer",
                "description": "$180,000",
            }
        )
        is not None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "CSR | Acme",
                "url": "https://acme.com/csr",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Purpose | Acme",
                "url": "https://acme.com/purpose",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "People | Acme",
                "url": "https://acme.com/people",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Purpose Engineer",
                "url": "https://jobs.example.com/job/purpose-engineer",
                "description": "$180,000",
            }
        )
        is not None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Ethics | Acme",
                "url": "https://acme.com/ethics",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Media Center | Acme",
                "url": "https://acme.com/media-center",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Environment | Acme",
                "url": "https://acme.com/environment",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Environment Engineer",
                "url": "https://jobs.example.com/job/environment-engineer",
                "description": "$180,000",
            }
        )
        is not None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Media Engineer",
                "url": "https://jobs.example.com/job/media-engineer",
                "description": "$180,000",
            }
        )
        is not None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Foundation | Acme",
                "url": "https://acme.com/foundation",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Giving | Acme",
                "url": "https://acme.com/giving",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Philanthropy | Acme",
                "url": "https://acme.com/philanthropy",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Foundation Engineer",
                "url": "https://jobs.example.com/job/foundation-engineer",
                "description": "$180,000",
            }
        )
        is not None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Citizenship | Acme",
                "url": "https://acme.com/citizenship",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Charity | Acme",
                "url": "https://acme.com/charity",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Charity Engineer",
                "url": "https://jobs.example.com/job/charity-engineer",
                "description": "$180,000",
            }
        )
        is not None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Responsibility | Acme",
                "url": "https://acme.com/responsibility",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Responsibility Engineer",
                "url": "https://jobs.example.com/job/responsibility-engineer",
                "description": "$180,000",
            }
        )
        is not None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Platform Team Engineer",
                "url": "https://jobs.example.com/job/platform-team-engineer",
                "description": "$180,000",
            }
        )
        is not None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "University Recruiting Coordinator",
                "url": "https://jobs.example.com/job/university-recruiting-coordinator",
                "description": "$180,000",
            }
        )
        is not None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Early Career Software Engineer",
                "url": "https://jobs.example.com/job/early-career-software-engineer",
                "description": "$180,000",
            }
        )
        is not None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Campus Recruiter",
                "url": "https://jobs.example.com/job/campus-recruiter",
                "description": "$180,000",
            }
        )
        is not None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Software Engineering Intern",
                "url": "https://acme.com/campus-recruiting/intern",
                "description": "$80,000",
            }
        )
        is not None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Software Engineering Intern",
                "url": "https://acme.com/internships/ml-intern",
                "description": "$80,000",
            }
        )
        is not None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Software Engineer",
                "url": "https://acme.com/hiring/senior-engineer",
                "description": "$180,000",
            }
        )
        is not None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Home | Grants.gov",
                "url": "https://www.grants.gov/",
                "description": "Find grants",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "AI/ML federal funding",
                "url": "https://nondilute.com/category/aiml/",
                "description": "52 open in 2026",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Search | Simpler.Grants.gov",
                "url": "https://www.grants.gov/search-grants/?keywords=intelligence",
                "description": "",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Hire a Freelance Machine Learning Engineer — No Agency Fees",
                "url": "https://remoteai.io/v2/freelance/machine-learning-engineers",
                "description": "Browse freelance ML engineers.",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Hire Machine Learning Engineers — Contract & C2C | Gain America",
                "url": "https://gainam.com/hire-machine-learning-engineers",
                "description": "without $500K comp",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "AI/ML Engineer - Freelance Job in AI & Machine Learning - Upwork",
                "url": "https://www.upwork.com/freelance-jobs/apply/Engineer_~022084959075748613623/",
                "description": "Senior Machine Learning Engineer contract.",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Machine Learning Engineering freelancers - contra.com",
                "url": "https://contra.com/hire/ml-engineers",
                "description": "$35-$100/hr",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "ML Engineer - Lemon.io",
                "url": "https://lemon.io/for-developers/ml-engineer-jobs/",
                "description": "ML Engineer on an oncology KOL analytics backend $35-$100/hr",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Ilias - Senior Machine Learning Engineer expert on Lemon.io",
                "url": "https://magic.lemon.io/share/ilias-s-gabgcvgom",
                "description": "",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Senior AI/ML Developer : Remote : Contract - Corp to Corp",
                "url": "https://corptocorp.org/senior-ai-ml-developer-remote-contract/",
                "description": "",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Senior Machine Learning (ML) Engineer - Freelance [Remote]",
                "url": "https://www.karkidi.com/job-details/76760-senior-machine-learning-ml-engineer-freelance-remote-job",
                "description": "Braintrust $80 - $100 / Hour. Posted on: 17 Apr 2024",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Senior Machine Learning (ML) Engineer - Freelance [Remote]",
                "url": "https://www.jobleads.com/us/job/senior-machine-learning-ml-engineer-freelance-remote-job",
                "description": "",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Senior Machine Learning Engineer",
                "url": "https://www.glassdoor.com/Job/remote-us-machine-learning-engineer-jobs-SRCH_IL.0,9_IS1_KO10,36.htm",
                "description": "$160K–$240K",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Senior Machine Learning Engineer",
                "url": "https://www.remoterocketship.com/company/acme/jobs/senior-ml-engineer",
                "description": "$160k remote",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Senior Machine Learning Engineer",
                "url": "https://migratemate.co/jobs/senior-machine-learning-engineer",
                "description": "United States $180k",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Senior Machine Learning Engineer",
                "url": "https://www.builtin.com/jobs/remote/ml",
                "description": "$160k–$200k",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Senior Machine Learning Engineer",
                "url": "https://www.ziprecruiter.com/Jobs/Senior-Machine-Learning-Engineer",
                "description": "$160k–$200k",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Senior Machine Learning Engineer - Acme",
                "url": "https://www.ziprecruiter.com/c/Acme/Job/Senior-Machine-Learning-Engineer/-in-San-Francisco,CA",
                "description": "$180,000 - $220,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Staff Machine Learning Engineer Job Description, Salary & Career Outlook",
                "url": "https://jobdescription.org/jobs/artificial-intelligence/staff-machine-learning-engineer",
                "description": "Staff Machine Learning Engineer salary ($195K–$310K)",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "AI Engineer",
                "url": "https://ai.engineer/jobs",
                "description": "$200k–$250k",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Responsibilities: Staff Machine Learning Engineer | Remotely",
                "url": "https://www.remotely.works/blog/what-are-the-responsibilities-of-a-staff-machine-learning",
                "description": "",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Senior Machine Learning Engineer",
                "url": "https://www.peopleinai.com/job/senior-machine-learning-engineer-9",
                "description": "California $200,000 - $300,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Senior Machine Learning Engineer Remote (United States)",
                "url": "https://7seventy.net/7job/759-lkdn-649-remote-united-states-senior-machine-learning-engineer",
                "description": "Compensation:$180,000 – $250,000 per year",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Senior Machine Learning Engineer | High Paying Remote Job",
                "url": "https://globalcareer.io/remote/jobs/senior-machine-learning-engineer/",
                "description": "$180,000 - $210,000 / year",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Senior Machine Learning Engineer at Rebtel",
                "url": "https://www.visa-hunt.com/jobs/472ad0f1815d89a1",
                "description": "Stockholm",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Senior Machine Learning Engineer at Clearview AI",
                "url": "https://dailyremote.com/remote-job/senior-machine-learning-engineer-5211999",
                "description": "$180k remote",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "WellRithms, Inc. hiring Senior AI/ML Engineer in Portland, OR | LinkedIn",
                "url": "https://www.linkedin.com/jobs/view/4459896965",
                "description": "Portland, OR $125,000.00 - $165,000.00",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Machine Learning Engineer Salaries by Country 2025-2026",
                "url": "https://optiveum.com/articles/machine-learning-engineer-salaries-by-country/",
                "description": "US $240,000 - $500,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Machine Learning Engineer Salary in Switzerland - SalaryExpert",
                "url": "https://www.salaryexpert.com/salary/job/machine-learning-engineer/switzerland",
                "description": "CHF 91'052",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Software Engineer at OpenAI",
                "url": "https://www.levels.fyi/companies/openai/salaries/software-engineer",
                "description": "$200,000 - $400,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Machine Learning Engineer",
                "url": "https://www.payscale.com/research/US/Job=Machine_Learning_Engineer/Salary",
                "description": "$150,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Machine Learning Engineer Salary",
                "url": "https://www.salary.com/research/salary/alternate/machine-learning-engineer-salary",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Machine Learning Engineer",
                "url": "https://www.acme.com/salaries/machine-learning-engineer",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Machine Learning Engineer: Average Salary & Pay Trends 2026",
                "url": "https://www.glassdoor.com/Salaries/switzerland-machine-learning-engineer-salary-SRCH_IL.0,11_IN226_KO12,37.htm",
                "description": "$120,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Senior Machine Learning Engineer | Cloudflare | Hybrid | July 2026",
                "url": "https://jobera.com/job/cloudflare-senior-machine-learning-engineer-6803af37/",
                "description": "$262k–$379k",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Senior Machine Learning Engineer, Search & Index - jobright.ai",
                "url": "https://jobright.ai/jobs/info/6a96d484455eaf6a08c18b9a",
                "description": "$312K/yr - $351K/yr",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Remote Senior Machine Learning Engineer at Greenhouse",
                "url": "https://remoteok.com/remote-jobs/remote-senior-machine-learning-engineer-greenhouse-1129790",
                "description": "$80k – $150k",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Staff Software Engineer, Machine Learning at SmarterDx",
                "url": "https://www.opentoworkremote.com/view/1470936",
                "description": "$230,000 - $250,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Staff ML Engineer at Cloudbeds",
                "url": "https://www.bilingualjobs.io/jobs/staff-ml-engineer-cloudbeds-greenhouse-cloud",
                "description": "Greenhouse",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "John Behling - Staff Engineer, Applied Machine Learning at Greenhouse",
                "url": "https://www.linkedin.com/in/john-behling-b75ba393",
                "description": "Greenhouse",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Staff Machine Learning Platform Engineer at Faire",
                "url": "https://jobquip.com/en/jobs/external-greenhouse-faire-careers-greenhouse-faire-staff-machine-learning-platform-engineer-54",
                "description": "Greenhouse",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Retrieve a Lead - developers.comeet.com",
                "url": "https://developers.comeet.com/reference/retrieve-a-lead",
                "description": "API",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "HR trends and what they mean: The AI job title | Personio",
                "url": "https://www.personio.com/blog/this-week-in-hr-ai-job-titles/",
                "description": "Blog",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Software Engineer",
                "url": "https://wellfound.com/role/l/software-engineer/united-states",
                "description": "$185k – $218k",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Software Engineer",
                "url": "https://wellfound.com/jobs?role=software-engineer",
                "description": "$185k – $218k",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "2026 Machine Learning Engineer Salary Guide: Insights for Senior Engineers",
                "url": "https://motionrecruitment.com/it-salary/machine-learning",
                "description": "$120,000 - $200,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Machine Learning Engineer Salary 2026: $108,500 Median Pay",
                "url": "https://salarybyrole.com/role/machine-learning-engineer",
                "description": "$108,500",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Senior Machine Learning Engineer H-1B Visa Salary Data",
                "url": "https://www.h1bscope.com/jobs/senior-machine-learning-engineer/",
                "description": "$180,000 median salary",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Compensation For AI Employees Is Skyrocketing - Forbes",
                "url": "https://www.forbes.com/sites/allbusiness/2026/01/07/compensation-for-ai-employees-is-skyrocketing/",
                "description": "$180,000-$350,000+",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Machine Learning Engineer Salary: How Much Can You Make?",
                "url": "https://www.coursera.org/articles/machine-learning-engineer-salary",
                "description": "$180,000",
            }
        )
        is None
    )
    kept_listing = _heuristic_opportunity(
        {
            "title": "Senior Machine Learning Engineer",
            "url": "https://www.glassdoor.com/job-listing/senior-machine-learning-engineer-acme-JV_IC1147401_KO0,32.htm",
            "description": "$180k–$220k",
        }
    )
    assert kept_listing is None
    kept_builtin = _heuristic_opportunity(
        {
            "title": "Senior Machine Learning Engineer",
            "url": "https://www.builtin.com/job/senior-machine-learning-engineer/12345",
            "description": "$180k–$220k",
        }
    )
    assert kept_builtin is not None
    kept_wellfound = _heuristic_opportunity(
        {
            "title": "IT Security Administrator at Bitwarden",
            "url": "https://wellfound.com/jobs/4335648-it-security-administrator",
            "description": "$115,000 - $145,000",
        }
    )
    assert kept_wellfound is not None
    assert kept_wellfound.pay_high == 145_000
    assert (
        _heuristic_opportunity(
            {
                "title": "Bitwarden",
                "url": "https://wellfound.com/company/bitwarden",
                "description": "$115,000 - $145,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Machine Learning Engineer Salary and Equity in 2026",
                "url": "https://wellfound.com/hiring-data/r/machine-learning-engineer-2",
                "description": "$180,000",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Machine Learning Engineer salary in US",
                "url": "https://builtin.com/salaries/us/machine-learning-engineer",
                "description": "$151,800",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Machine Learning Engineer Salary in San Francisco",
                "url": "https://www.builtinsf.com/salaries/us/san-francisco/machine-learning-engineer",
                "description": "$151,800",
            }
        )
        is None
    )
    kept_city = _heuristic_opportunity(
        {
            "title": "Staff Machine Learning Engineer",
            "url": "https://www.builtinsf.com/job/staff-machine-learning-engineer/7823375",
            "description": "$180k–$220k",
        }
    )
    assert kept_city is not None
    assert kept_city.pay_high == 220_000
    assert (
        _heuristic_opportunity(
            {
                "title": "Machine Learning Engineer - Built In",
                "url": "https://builtin.com/learn/careers/machine-learning-engineer",
                "description": "Career guide",
            }
        )
        is None
    )
    amgen = _heuristic_opportunity(
        {
            "title": "Senior Machine Learning Engineer Jobs at Amgen in United States - Remote",
            "url": "https://careers.amgen.com/en/job/washington-d-c/senior-machine-learning-engineer/87/99808047504",
            "description": "Remote",
        }
    )
    assert amgen is not None
    from src.engine import _html_is_index, _is_index_page

    assert not _is_index_page(
        {
            "url": "https://job-boards.greenhouse.io/grafanalabs/jobs/1",
            "title": "Jobs at Grafana Labs",
            "description": "",
        }
    )
    assert _html_is_index(
        "<title>Jobs at Grafana Labs</title><p>Current openings</p>",
        "https://job-boards.greenhouse.io/grafanalabs/jobs/1",
    )
    assert _html_is_index(
        "<title>Working in Artificial Intelligence | People in AI</title>"
        "<p>California $200,000 - $300,000</p>",
        "https://www.peopleinai.com/job/senior-machine-learning-engineer-9",
    )
    assert _html_is_index(
        "<title>WellRithms, Inc. hiring Senior AI/ML Engineer in Portland, OR | LinkedIn</title>"
        "<p>Boomerang Healthcare Portland, OR $125,000.00 - $165,000.00</p>",
        "https://www.linkedin.com/jobs/view/4459896965",
    )
    assert _html_is_index(
        "<title>Senior Machine Learning Engineer - Acme</title>"
        "<p>$180,000 - $220,000 a year</p>",
        "https://www.indeed.com/viewjob?jk=abc123def456",
    )
    assert _html_is_index(
        "<title>Senior Machine Learning Engineer - Acme</title>"
        "<p>$180k–$220k</p>",
        "https://www.glassdoor.com/job-listing/senior-machine-learning-engineer-acme-JV_IC1147401_KO0,32.htm",
    )
    assert _html_is_index(
        "<title>John Behling - Staff Engineer, Applied Machine Learning at Greenhouse</title>",
        "https://www.linkedin.com/in/john-behling-b75ba393",
    )
    assert _html_is_index(
        "<title>Retrieve a Lead</title>",
        "https://developers.comeet.com/reference/retrieve-a-lead",
    )
    assert not _html_is_index(
        "<title>Generative AI Pipeline Engineer (Tech Lead)</title>",
        "https://www.comeet.com/jobs/capslock/59.001/generative-ai-pipeline-engineer-tech-lead/60.F60-8B.403",
    )
    assert _html_is_index(
        "<title>Senior Machine Learning Engineer | Cloudflare | Hybrid</title>"
        "<p>Agent Harness - Meta Factory$262k–$379k/yr</p>",
        "https://jobera.com/job/cloudflare-senior-machine-learning-engineer-6803af37/",
    )
    assert _html_is_index(
        "<p>$312K/yr - $351K/yr</p>",
        "https://jobright.ai/jobs/info/6a96d484455eaf6a08c18b9a",
    )
    assert _html_is_index(
        "<title>Remote Senior Machine Learning Engineer at Greenhouse</title>"
        "<p>$80,000 – $150,000</p>",
        "https://remoteok.com/remote-jobs/remote-senior-machine-learning-engineer-greenhouse-1129790",
    )
    assert _html_is_index(
        "<title>Staff Software Engineer, Machine Learning at SmarterDx</title>"
        "<p>$230,000 - $250,000</p>",
        "https://www.opentoworkremote.com/view/1470936",
    )
    assert _html_is_index(
        "<title>Staff ML Engineer at Cloudbeds</title><p>Remote</p>",
        "https://www.bilingualjobs.io/jobs/staff-ml-engineer-cloudbeds-greenhouse-cloud",
    )
    assert _html_is_index(
        "<title>Software Engineer</title><p>$185k – $218k</p>",
        "https://wellfound.com/role/l/software-engineer/united-states",
    )
    assert _html_is_index(
        "<title>Software Engineer</title><p>$110k – $200k</p>",
        "https://wellfound.com/jobs?role=software-engineer",
    )
    assert _html_is_index(
        "<title>Bitwarden</title><p>$115,000 - $145,000</p>",
        "https://wellfound.com/company/bitwarden",
    )
    assert _html_is_index(
        "<title>Machine Learning</title><p>$120,000 - $200,000</p>",
        "https://motionrecruitment.com/it-salary/machine-learning",
    )
    assert _html_is_index(
        "<title>Machine Learning Engineer Salary 2026: $108,500 Median Pay</title>"
        "<p>$108,500</p>",
        "https://salarybyrole.com/role/machine-learning-engineer",
    )
    assert not _html_is_index(
        "<title>IT Security Administrator at Bitwarden</title><p>$115,000 - $145,000</p>",
        "https://wellfound.com/jobs/4335648-it-security-administrator",
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Staff Software Engineer",
                "url": "https://www.greenhouse.com/careers",
                "description": "Greenhouse $150,000 - $220,000",
            }
        )
        is None
    )
    assert _html_is_index(
        "<title>Staff Software Engineer</title>"
        '<script type="application/ld+json">{"@type":"JobPosting","title":"Staff Software Engineer"}</script>',
        "https://www.greenhouse.com/careers",
    )
    assert _is_index_page(
        {
            "url": "https://www.greenhouse.com/careers",
            "title": "Staff Software Engineer",
            "description": "",
        }
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Careers at Scale AI | Build the Future of AI | Scale AI",
                "url": "https://scale.com/careers",
                "description": "Staff Machine Learning Engineer",
            }
        )
        is None
    )
    assert _html_is_index(
        "<title>Staff Machine Learning Engineer</title>"
        '<script type="application/ld+json">{"@type":"JobPosting"}</script>',
        "https://scale.com/careers",
    )
    assert _html_is_index(
        "<title>Software Engineer at OpenAI</title><p>$200,000 - $400,000</p>",
        "https://www.levels.fyi/companies/openai/salaries/software-engineer",
    )
    assert _html_is_index(
        "<title>Machine Learning Engineer</title><p>$180,000</p>",
        "https://www.acme.com/salaries/machine-learning-engineer",
    )
    assert not _html_is_index(
        "<title>Salary Transparency Engineer</title><p>$180,000 a year</p>",
        "https://jobs.example.com/job/salary-transparency-engineer",
    )
    assert _html_is_index(
        "<title>Staff Software Engineer</title>",
        "https://www.lever.co/careers",
    )
    assert _html_is_index(
        "<title>Machine Learning Engineering freelancers</title>"
        "<p>$50,000</p>",
        "https://contra.com/hire/ml-engineers",
    )
    assert _html_is_index(
        "<title>Open Positions | Stripe</title><p>$180,000 - $270,000</p>",
        "https://stripe.com/jobs/open-positions",
    )
    assert _html_is_index(
        "<title>Software Engineer</title><p>$180,000</p>",
        "https://acme.com/careers/open-positions",
    )
    assert _html_is_index(
        "<title>Open Roles | Acme</title><p>$180,000</p>",
        "https://acme.com/teams/engineering",
    )
    assert not _html_is_index(
        "<title>Open Position: Software Engineer</title><p>$180,000</p>",
        "https://jobs.example.com/job/open-position-software-engineer",
    )
    assert not _html_is_index(
        "<title>Senior Engineer</title><p>$180,000</p>",
        "https://acme.com/open-positions/senior-engineer",
    )
    assert _html_is_index(
        "<title>Career Opportunities | Acme</title><p>$180,000 - $270,000</p>",
        "https://acme.com/career-opportunities",
    )
    assert _html_is_index(
        "<title>Software Engineer</title><p>$180,000</p>",
        "https://acme.com/job-openings",
    )
    assert _html_is_index(
        "<title>Careers at Acme</title><p>$200,000</p>",
        "https://acme.com/about/careers-team",
    )
    assert not _html_is_index(
        "<title>Job Opening: Software Engineer</title><p>$180,000</p>",
        "https://jobs.example.com/job/job-opening-software-engineer",
    )
    assert not _html_is_index(
        "<title>Senior Engineer</title><p>$180,000</p>",
        "https://acme.com/job-openings/senior-engineer",
    )
    assert _html_is_index(
        "<title>Join Our Team | Acme</title><p>$180,000</p>",
        "https://acme.com/join-our-team",
    )
    assert _html_is_index(
        "<title>Software Engineer</title><p>$180,000</p>",
        "https://acme.com/work-with-us",
    )
    assert _html_is_index(
        "<title>We're Hiring | Acme</title><p>$180,000</p>",
        "https://acme.com/were-hiring",
    )
    assert _html_is_index(
        "<title>Opportunities | Acme</title><p>$200,000</p>",
        "https://acme.com/opportunities",
    )
    assert not _html_is_index(
        "<title>Join Our Team as a Software Engineer</title><p>$180,000</p>",
        "https://jobs.example.com/job/join-our-team-software-engineer",
    )
    assert not _html_is_index(
        "<title>Senior Engineer</title><p>$180,000</p>",
        "https://acme.com/join-our-team/senior-engineer",
    )
    assert _html_is_index(
        "<title>Job Vacancies | Acme</title><p>$180,000</p>",
        "https://acme.com/vacancies",
    )
    assert _html_is_index(
        "<title>Available Positions</title><p>$180,000</p>",
        "https://acme.com/available-positions",
    )
    assert _html_is_index(
        "<title>Hiring | Acme</title><p>$180,000</p>",
        "https://acme.com/hiring",
    )
    assert _html_is_index(
        "<title>Explore Careers</title><p>$180,000</p>",
        "https://acme.com/explore-careers",
    )
    assert not _html_is_index(
        "<title>Hiring Manager</title><p>$180,000</p>",
        "https://jobs.example.com/job/hiring-manager",
    )
    assert not _html_is_index(
        "<title>Senior Engineer</title><p>$180,000</p>",
        "https://acme.com/hiring/senior-engineer",
    )
    assert _html_is_index(
        "<title>Life at Acme</title><p>$180,000</p>",
        "https://acme.com/life-at-acme",
    )
    assert _html_is_index(
        "<title>Meet the Team | Acme</title><p>$180,000</p>",
        "https://acme.com/meet-the-team",
    )
    assert _html_is_index(
        "<title>Engineer</title><p>$180,000</p>",
        "https://acme.com/our-people",
    )
    assert not _html_is_index(
        "<title>Our Team Lead</title><p>$180,000</p>",
        "https://jobs.example.com/job/our-team-lead",
    )
    assert _html_is_index(
        "<title>Team | Acme</title><p>$180,000</p>",
        "https://acme.com/team",
    )
    assert _html_is_index(
        "<title>Why Acme</title><p>$180,000</p>",
        "https://acme.com/about",
    )
    assert _html_is_index(
        "<title>Internships | Acme</title><p>$180,000</p>",
        "https://acme.com/internships",
    )
    assert _html_is_index(
        "<title>University Recruiting</title><p>$180,000</p>",
        "https://acme.com/university-recruiting",
    )
    assert _html_is_index(
        "<title>Campus Recruiting | Acme</title><p>$180,000</p>",
        "https://acme.com/campus-recruiting",
    )
    assert _html_is_index(
        "<title>Early Careers | Acme</title><p>$180,000</p>",
        "https://acme.com/early-careers",
    )
    assert _html_is_index(
        "<title>Student Programs</title><p>$180,000</p>",
        "https://acme.com/student-programs",
    )
    assert _html_is_index(
        "<title>Graduate Programs | Acme</title><p>$180,000</p>",
        "https://acme.com/graduate-programs",
    )
    assert _html_is_index(
        "<title>Job Search | Acme</title><p>$180,000</p>",
        "https://acme.com/job-search",
    )
    assert not _html_is_index(
        "<title>Search Engineer</title><p>$180,000</p>",
        "https://jobs.example.com/job/search-engineer",
    )
    assert not _html_is_index(
        "<title>Senior Engineer</title><p>$180,000</p>",
        "https://acme.com/job-search/senior-engineer",
    )
    assert _html_is_index(
        "<title>Careers | Acme</title><p>$180,000</p>",
        "https://acme.com/about/careers-overview",
    )
    assert _html_is_index(
        "<title>Benefits | Acme</title><p>$180,000</p>",
        "https://acme.com/benefits",
    )
    assert not _html_is_index(
        "<title>Benefits Analyst</title><p>$180,000</p>",
        "https://jobs.example.com/job/benefits-analyst",
    )
    assert _html_is_index(
        "<title>Culture | Acme</title><p>$180,000</p>",
        "https://acme.com/culture",
    )
    assert _html_is_index(
        "<title>Leadership | Acme</title><p>$180,000</p>",
        "https://acme.com/leadership",
    )
    assert not _html_is_index(
        "<title>Culture Engineer</title><p>$180,000</p>",
        "https://jobs.example.com/job/culture-engineer",
    )
    assert not _html_is_index(
        "<title>Leadership Development Program</title><p>$180,000</p>",
        "https://jobs.example.com/job/ldp",
    )
    assert _html_is_index(
        "<title>About Us | Acme</title><p>$180,000</p>",
        "https://acme.com/about-us",
    )
    assert _html_is_index(
        "<title>Our Values | Acme</title><p>$180,000</p>",
        "https://acme.com/our-values",
    )
    assert _html_is_index(
        "<title>Locations | Acme</title><p>$180,000</p>",
        "https://acme.com/locations",
    )
    assert not _html_is_index(
        "<title>Values Engineer</title><p>$180,000</p>",
        "https://jobs.example.com/job/values-engineer",
    )
    assert not _html_is_index(
        "<title>About this Role</title><p>$180,000</p>",
        "https://jobs.example.com/job/about-this-role",
    )
    assert _html_is_index(
        "<title>Diversity | Acme</title><p>$180,000</p>",
        "https://acme.com/diversity",
    )
    assert _html_is_index(
        "<title>DEI | Acme</title><p>$180,000</p>",
        "https://acme.com/dei",
    )
    assert not _html_is_index(
        "<title>Diversity Engineer</title><p>$180,000</p>",
        "https://jobs.example.com/job/diversity-engineer",
    )
    assert _html_is_index(
        "<title>Our Story | Acme</title><p>$180,000</p>",
        "https://acme.com/our-story",
    )
    assert _html_is_index(
        "<title>FAQs | Acme</title><p>$180,000</p>",
        "https://acme.com/faqs",
    )
    assert not _html_is_index(
        "<title>Story Engineer</title><p>$180,000</p>",
        "https://jobs.example.com/job/story-engineer",
    )
    assert _html_is_index(
        "<title>News | Acme</title><p>$180,000</p>",
        "https://acme.com/news",
    )
    assert not _html_is_index(
        "<title>News Engineer</title><p>$180,000</p>",
        "https://jobs.example.com/job/news-engineer",
    )
    assert not _html_is_index(
        "<title>Senior Engineer</title><p>$180,000</p>",
        "https://acme.com/news/senior-engineer",
    )
    assert _html_is_index(
        "<title>Newsroom | Acme</title><p>$180,000</p>",
        "https://acme.com/newsroom",
    )
    assert _html_is_index(
        "<title>Investors | Acme</title><p>$180,000</p>",
        "https://acme.com/investors",
    )
    assert not _html_is_index(
        "<title>Investors Engineer</title><p>$180,000</p>",
        "https://jobs.example.com/job/investors-engineer",
    )
    assert not _html_is_index(
        "<title>Senior Engineer</title><p>$180,000</p>",
        "https://acme.com/investors/senior-engineer",
    )
    assert not _html_is_index(
        "<title>Senior Engineer</title><p>$180,000</p>",
        "https://acme.com/newsroom/senior-engineer",
    )
    assert _html_is_index(
        "<title>Sustainability | Acme</title><p>$180,000</p>",
        "https://acme.com/sustainability",
    )
    assert not _html_is_index(
        "<title>Sustainability Engineer</title><p>$180,000</p>",
        "https://jobs.example.com/job/sustainability-engineer",
    )
    assert not _html_is_index(
        "<title>Senior Engineer</title><p>$180,000</p>",
        "https://acme.com/sustainability/senior-engineer",
    )
    assert _html_is_index(
        "<title>ESG | Acme</title><p>$180,000</p>",
        "https://acme.com/esg",
    )
    assert _html_is_index(
        "<title>Impact | Acme</title><p>$180,000</p>",
        "https://acme.com/impact",
    )
    assert _html_is_index(
        "<title>Community | Acme</title><p>$180,000</p>",
        "https://acme.com/community",
    )
    assert not _html_is_index(
        "<title>Impact Engineer</title><p>$180,000</p>",
        "https://jobs.example.com/job/impact-engineer",
    )
    assert not _html_is_index(
        "<title>Senior Engineer</title><p>$180,000</p>",
        "https://acme.com/impact/senior-engineer",
    )
    assert _html_is_index(
        "<title>CSR | Acme</title><p>$180,000</p>",
        "https://acme.com/csr",
    )
    assert _html_is_index(
        "<title>Purpose | Acme</title><p>$180,000</p>",
        "https://acme.com/purpose",
    )
    assert _html_is_index(
        "<title>People | Acme</title><p>$180,000</p>",
        "https://acme.com/people",
    )
    assert not _html_is_index(
        "<title>Purpose Engineer</title><p>$180,000</p>",
        "https://jobs.example.com/job/purpose-engineer",
    )
    assert not _html_is_index(
        "<title>Senior Engineer</title><p>$180,000</p>",
        "https://acme.com/purpose/senior-engineer",
    )
    assert _html_is_index(
        "<title>Ethics | Acme</title><p>$180,000</p>",
        "https://acme.com/ethics",
    )
    assert _html_is_index(
        "<title>Media Center | Acme</title><p>$180,000</p>",
        "https://acme.com/media-center",
    )
    assert _html_is_index(
        "<title>Environment | Acme</title><p>$180,000</p>",
        "https://acme.com/environment",
    )
    assert not _html_is_index(
        "<title>Environment Engineer</title><p>$180,000</p>",
        "https://jobs.example.com/job/environment-engineer",
    )
    assert not _html_is_index(
        "<title>Media Engineer</title><p>$180,000</p>",
        "https://jobs.example.com/job/media-engineer",
    )
    assert not _html_is_index(
        "<title>Senior Engineer</title><p>$180,000</p>",
        "https://acme.com/environment/senior-engineer",
    )
    assert _html_is_index(
        "<title>Foundation | Acme</title><p>$180,000</p>",
        "https://acme.com/foundation",
    )
    assert _html_is_index(
        "<title>Giving | Acme</title><p>$180,000</p>",
        "https://acme.com/giving",
    )
    assert _html_is_index(
        "<title>Philanthropy | Acme</title><p>$180,000</p>",
        "https://acme.com/philanthropy",
    )
    assert not _html_is_index(
        "<title>Foundation Engineer</title><p>$180,000</p>",
        "https://jobs.example.com/job/foundation-engineer",
    )
    assert not _html_is_index(
        "<title>Senior Engineer</title><p>$180,000</p>",
        "https://acme.com/foundation/senior-engineer",
    )
    assert _html_is_index(
        "<title>Citizenship | Acme</title><p>$180,000</p>",
        "https://acme.com/citizenship",
    )
    assert _html_is_index(
        "<title>Charity | Acme</title><p>$180,000</p>",
        "https://acme.com/charity",
    )
    assert not _html_is_index(
        "<title>Charity Engineer</title><p>$180,000</p>",
        "https://jobs.example.com/job/charity-engineer",
    )
    assert not _html_is_index(
        "<title>Senior Engineer</title><p>$180,000</p>",
        "https://acme.com/charity/senior-engineer",
    )
    assert _html_is_index(
        "<title>Responsibility | Acme</title><p>$180,000</p>",
        "https://acme.com/responsibility",
    )
    assert not _html_is_index(
        "<title>Responsibility Engineer</title><p>$180,000</p>",
        "https://jobs.example.com/job/responsibility-engineer",
    )
    assert not _html_is_index(
        "<title>Senior Engineer</title><p>$180,000</p>",
        "https://acme.com/responsibility/senior-engineer",
    )
    assert not _html_is_index(
        "<title>Senior Engineer</title><p>$180,000</p>",
        "https://acme.com/faq/senior-engineer",
    )
    assert not _html_is_index(
        "<title>Senior Engineer</title><p>$180,000</p>",
        "https://acme.com/diversity/senior-engineer",
    )
    assert not _html_is_index(
        "<title>Senior Engineer</title><p>$180,000</p>",
        "https://acme.com/about/senior-engineer",
    )
    assert not _html_is_index(
        "<title>Early Career Software Engineer</title><p>$180,000</p>",
        "https://jobs.example.com/job/early-career-software-engineer",
    )
    assert not _html_is_index(
        "<title>Campus Recruiter</title><p>$180,000</p>",
        "https://jobs.example.com/job/campus-recruiter",
    )
    assert not _html_is_index(
        "<title>Software Engineering Intern</title><p>$80,000</p>",
        "https://acme.com/campus-recruiting/intern",
    )
    assert _html_is_index(
        "<title>Software Engineer</title><p>$180,000</p>",
        "https://acme.com/internships",
    )
    assert not _html_is_index(
        "<title>Platform Team Engineer</title><p>$180,000</p>",
        "https://jobs.example.com/job/platform-team-engineer",
    )
    assert not _html_is_index(
        "<title>University Recruiting Coordinator</title><p>$180,000</p>",
        "https://jobs.example.com/job/university-recruiting-coordinator",
    )
    assert not _html_is_index(
        "<title>Software Engineering Intern</title><p>$80,000</p>",
        "https://acme.com/internships/ml-intern",
    )
    assert not _html_is_index(
        "<title>Senior Machine Learning Engineer - Freelance</title>",
        "https://jobs.lever.co/acme/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    )
    assert not _html_is_index(
        "<title>Staff Machine Learning Engineer</title>",
        "https://job-boards.greenhouse.io/reddit/jobs/7747244",
    )
    assert _html_is_index(
        "<title>Staff Machine Learning Engineer - Edge AI</title>",
        "https://www.samsara.com/careers?gh_jid=7266357",
    ) is False
    assert (
        _heuristic_opportunity(
            {
                "title": "Staff AI Engineer (d/f/m) - Internal AI - Personio",
                "url": "https://www.personio.com/careers/3872de70-5678-44be-90fd-475541abd6f4/apply/",
                "description": "Personio",
            }
        )
        is None
    )
    assert _html_is_index(
        "<title>Staff AI Engineer</title>",
        "https://www.personio.com/careers/3872de70-5678-44be-90fd-475541abd6f4/apply/",
    )
    assert not _is_index_page(
        {
            "url": "https://inbrain-neuroelectronics.jobs.personio.com/job/2749890",
            "title": "Machine Learning Operations Architect",
            "description": "",
        }
    )
    assert not _html_is_index(
        "<title>Search Engineer</title><p>$180,000</p>",
        "https://jobs.example.com/job/search-engineer",
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Search Engineer",
                "url": "https://jobs.example.com/job/search-platform",
                "description": "$180,000",
            }
        )
        is not None
    )
    assert _is_index_page(
        {
            "url": "https://jobs.example.com/en/search",
            "title": "Search Engineer",
            "description": "",
        }
    )
    assert not _is_index_page(
        {
            "url": "https://job-boards.eu.greenhouse.io/overstory/jobs/4411330101",
            "title": "Jobs at Overstory",
            "description": "",
        }
    )
    assert _is_index_page(
        {"url": "https://jobs.ashbyhq.com/acme", "title": "Jobs", "description": ""}
    )
    assert _is_index_page(
        {"url": "https://jobs.ashbyhq.com/webai", "title": "webAI", "description": ""}
    )
    assert _is_index_page(
        {
            "url": "https://job-boards.greenhouse.io/reddit",
            "title": "Reddit",
            "description": "",
        }
    )
    assert not _is_index_page(
        {
            "url": "https://job-boards.greenhouse.io/reddit/jobs/7747244",
            "title": "Staff Machine Learning Engineer",
            "description": "",
        }
    )
    yelp = _heuristic_opportunity(
        {
            "title": "Careers at Yelp | Yelp Jobs",
            "url": "https://uscareers-yelp.icims.com/jobs/13815/senior-machine-learning-engineer/job",
            "description": "Remote United States",
        }
    )
    assert yelp is not None
    assert (
        _heuristic_opportunity(
            {
                "title": "Jobgether - Senior Machine Learning Engineer",
                "url": "https://jobs.lever.co/jobgether/dd9c2026-60c2-4c5f-b507-dc9d22cc68b9",
                "description": "This position is listed on behalf of a partner company.",
            }
        )
        is None
    )
    qonto = _heuristic_opportunity(
        {
            "title": "Qonto - Senior Machine Learning Engineer for AI Product",
            "url": "https://jobs.lever.co/qonto/471e0021-d630-4cd1-81c3-2fb2e9dc253c",
            "description": "",
        }
    )
    assert qonto is not None
    assert (
        _heuristic_opportunity(
            {
                "title": "Senior Machine Learning Engineer at ServiceNow",
                "url": "https://www.remotesource.com/jobs/yZVKfcilCyoKCJTGfDGhF-senior-machine-learning-engineer-at-servicenow",
                "description": "Canada Full-Time",
            }
        )
        is None
    )


def test_heuristic_stores_lever_job_url_not_apply():
    h = _heuristic_opportunity(
        {
            "title": "Provectus - Senior AI/ML Engineer (GenAI, AWS)",
            "url": "https://jobs.lever.co/provectus/0bf1decc-002c-4b0a-b97b-6407d2930fff/apply",
            "description": "",
        }
    )
    assert h.url == "https://jobs.lever.co/provectus/0bf1decc-002c-4b0a-b97b-6407d2930fff"


def test_search_all_drops_index_pages():
    engine = Engine()

    async def fake_brave(_query: str):
        return [
            {
                "title": "Jobs - Indeed",
                "url": "https://www.indeed.com/q-ml-jobs.html",
            },
            {
                "title": "Senior Machine Learning Engineer",
                "url": "https://www.glassdoor.com/Job/remote-us-machine-learning-engineer-jobs-SRCH_IL.0,9_IS1_KO10,36.htm",
            },
            {
                "title": "Senior Machine Learning Engineer",
                "url": "https://www.remoterocketship.com/company/acme/jobs/senior-ml-engineer",
            },
            {
                "title": "Senior Machine Learning Engineer",
                "url": "https://migratemate.co/jobs/senior-machine-learning-engineer",
            },
            {
                "title": "Senior Machine Learning Engineer",
                "url": "https://www.builtin.com/jobs/remote/ml",
            },
            {
                "title": "Senior Machine Learning Engineer",
                "url": "https://www.ziprecruiter.com/jobs-search?search=ml",
            },
            {
                "title": "Jobgether - Senior Machine Learning Engineer",
                "url": "https://jobs.lever.co/jobgether/dd9c2026-60c2-4c5f-b507-dc9d22cc68b9",
            },
            {
                "title": "Senior Machine Learning Engineer at ServiceNow",
                "url": "https://www.remotesource.com/jobs/abc-senior-machine-learning-engineer-at-servicenow",
            },
            {
                "title": "Remote Machine Learning",
                "url": "https://arc.dev/remote-jobs/machine-learning",
            },
            {
                "title": "Staff Machine Learning Engineer Job Description",
                "url": "https://jobdescription.org/jobs/artificial-intelligence/staff-machine-learning-engineer",
            },
            {
                "title": "AI Engineer",
                "url": "https://ai.engineer/jobs",
            },
            {
                "title": "Staff Machine Learning Engineer",
                "url": "https://www.remotely.works/blog/what-are-the-responsibilities-of-a-staff-machine-learning",
            },
            {
                "title": "Senior Machine Learning Engineer",
                "url": "https://www.peopleinai.com/job/senior-machine-learning-engineer-9",
            },
            {
                "title": "Senior Machine Learning Engineer Remote (United States)",
                "url": "https://7seventy.net/7job/759-lkdn-649-remote-united-states-senior-machine-learning-engineer",
            },
            {
                "title": "Senior Machine Learning Engineer | High Paying Remote Job",
                "url": "https://globalcareer.io/remote/jobs/senior-machine-learning-engineer/",
            },
            {
                "title": "Senior Machine Learning Engineer at Rebtel",
                "url": "https://www.visa-hunt.com/jobs/472ad0f1815d89a1",
            },
            {
                "title": "Senior Machine Learning Engineer at Clearview AI",
                "url": "https://dailyremote.com/remote-job/senior-machine-learning-engineer-5211999",
            },
            {
                "title": "WellRithms, Inc. hiring Senior AI/ML Engineer in Portland, OR | LinkedIn",
                "url": "https://www.linkedin.com/jobs/view/4459896965",
            },
            {
                "title": "Machine Learning Engineer Salaries by Country",
                "url": "https://optiveum.com/articles/machine-learning-engineer-salaries-by-country/",
            },
            {
                "title": "Machine Learning Engineer Salary in Switzerland",
                "url": "https://www.salaryexpert.com/salary/job/machine-learning-engineer/switzerland",
            },
            {
                "title": "Machine Learning Engineer Salaries",
                "url": "https://www.glassdoor.com/Salaries/machine-learning-engineer-salary-SRCH_KO0,25.htm",
            },
            {
                "title": "Senior Machine Learning Engineer | Cloudflare | Hybrid",
                "url": "https://jobera.com/job/cloudflare-senior-machine-learning-engineer-6803af37/",
            },
            {
                "title": "Senior Machine Learning Engineer, Search & Index",
                "url": "https://jobright.ai/jobs/info/6a96d484455eaf6a08c18b9a",
            },
            {"title": "Real role", "url": "https://jobs.example/ml"},
        ]

    async def fake_perplexity(_query: str):
        return []

    engine._search_brave = fake_brave
    engine._search_perplexity = fake_perplexity
    results = asyncio.run(engine._search_all("ml"))
    assert [r["url"] for r in results] == ["https://jobs.example/ml"]


def test_search_all_runs_site_angles_before_generic():
    engine = Engine()
    seen: list[str] = []

    async def fake_brave(query: str):
        seen.append(query)
        return [{"title": "R", "url": f"https://jobs.example/{len(seen)}"}]

    engine._search_brave = fake_brave
    asyncio.run(engine._search_all("ml"))
    generic = [q for q in seen if "site:" not in q]
    sites = [q for q in seen if "site:" in q]
    assert generic
    assert sites
    assert seen == sites + generic


def test_search_all_retries_empty_site_angles_after_generic():
    engine = Engine()
    seen: list[str] = []

    async def fake_brave(query: str):
        seen.append(query)
        if "ashbyhq.com" in query and seen.count(query) == 1:
            return []
        return [{"title": "R", "url": f"https://jobs.example/{len(seen)}"}]

    engine._search_brave = fake_brave
    results = asyncio.run(engine._search_all("ml"))
    ashby = "ml site:jobs.ashbyhq.com"
    assert seen.count(ashby) == 2
    assert seen.index(ashby) < seen.index("ml")
    assert seen[-1] == ashby
    assert "https://jobs.example/17" in [r["url"] for r in results]


def test_search_angles_omit_grants_and_equity_unless_asked():
    job = _search_angles("senior ML engineer remote")
    assert job == [
        "senior ML engineer remote",
        "senior ML engineer remote job hiring",
        "senior ML engineer remote freelance contract",
        "senior ML engineer remote site:greenhouse.io",
        "senior ML engineer remote site:jobs.lever.co",
        "senior ML engineer remote site:jobs.eu.lever.co",
        "senior ML engineer remote site:jobs.ashbyhq.com",
        "senior ML engineer remote site:jobs.workable.com",
        "senior ML engineer remote site:apply.workable.com",
        "senior ML engineer remote site:jobs.smartrecruiters.com",
        "senior ML engineer remote site:myworkdayjobs.com",
        "senior ML engineer remote site:icims.com",
        "senior ML engineer remote site:jobvite.com",
        "senior ML engineer remote site:teamtailor.com",
        "senior ML engineer remote site:personio.com",
        "senior ML engineer remote site:personio.de",
        "senior ML engineer remote site:recruitee.com",
        "senior ML engineer remote site:ats.rippling.com",
        "senior ML engineer remote site:breezy.hr",
        "senior ML engineer remote site:pinpointhq.com",
        "senior ML engineer remote site:comeet.com",
        "senior ML engineer remote site:bamboohr.com",
        "senior ML engineer remote site:applytojob.com",
        "senior ML engineer remote site:app.dover.com",
        "senior ML engineer remote site:jobs.gem.com",
        "senior ML engineer remote site:careers.walmart.com",
        "senior ML engineer remote site:jobs.apple.com",
        "senior ML engineer remote site:wellfound.com",
        "senior ML engineer remote site:builtin.com",
    ]
    assert _search_angles("ml site:example.com") == [
        "ml site:example.com",
        "ml site:example.com remote job hiring",
        "ml site:example.com freelance contract",
    ]
    grant = _search_angles("AI grant funding")
    assert "AI grant funding opportunity" in grant
    assert any("hiring" in q for q in grant)
    equity = _search_angles("startup cofounder")
    assert "startup cofounder equity" in equity


def test_find_ranks_and_limits_without_llm():
    engine = Engine()
    engine.openai = None

    async def fake_search(_query: str):
        return [
            {
                "title": "Low",
                "url": "https://a.example/low",
                "description": "",
                "pay": 100_000,
                "hours": 40,
                "remote": True,
            },
            {
                "title": "High",
                "url": "https://a.example/high",
                "description": "",
                "pay": 200_000,
                "hours": 20,
                "remote": True,
            },
            {
                "title": "Office",
                "url": "https://a.example/office",
                "description": "",
                "pay": 200_000,
                "hours": 40,
                "remote": False,
            },
            {"title": "dropped — no url", "pay": 999_999, "hours": 1},
        ]

    engine._search_all = fake_search
    ranked = asyncio.run(engine.find("ml", limit=2))
    assert [o.title for o in ranked] == ["High", "Office"]
    assert ranked[0].score() == 200.0
    assert ranked[1].score() == 70.0


def test_find_dedupes_same_title_keeps_higher_score():
    engine = Engine()
    engine.openai = None

    async def fake_search(_query: str):
        return [
            {
                "title": "Senior ML Engineer",
                "url": "https://board-a.example/1",
                "description": "$100k",
            },
            {
                "title": "Senior ML Engineer",
                "url": "https://board-b.example/2",
                "description": "$180k",
            },
            {
                "title": "Other Role $90k",
                "url": "https://board-c.example/3",
                "description": "",
            },
        ]

    engine._search_all = fake_search
    ranked = asyncio.run(engine.find("ml", limit=20))
    assert [o.title for o in ranked] == ["Senior ML Engineer", "Other Role $90k"]
    assert ranked[0].url == "https://board-b.example/2"
    assert ranked[0].pay_high == 180_000


def test_dedupe_keeps_same_title_at_different_companies():
    from src.engine import _dedupe_opportunities

    quilter_low = Opportunity(
        title="Senior ML Engineer @ Quilter",
        url="https://jobs.ashbyhq.com/quilter/low",
        company="Quilter",
        pay_high=100_000,
        hours_per_week=40,
    )
    quilter = Opportunity(
        title="Senior ML Engineer",
        url="https://jobs.ashbyhq.com/quilter/2b0f95cb-7c8b-4b62-8bcb-b9993344f2f1",
        company="Quilter",
        pay_low=180_000,
        pay_high=200_000,
        hours_per_week=40,
    )
    coral = Opportunity(
        title="Senior ML Engineer",
        url="https://jobs.ashbyhq.com/coralai/1ce17887-c305-4d77-a659-f75cf74bf8af",
        company="Coral AI",
    )
    ranked = sorted(
        [coral, quilter_low, quilter],
        key=lambda o: o.score(),
        reverse=True,
    )
    out = _dedupe_opportunities(ranked)
    assert [o.company for o in out] == ["Quilter", "Coral AI"]
    assert out[0].url == quilter.url
    assert out[0].pay_high == 200_000


def test_dedupe_collapses_team_suffix_syndication():
    from src.engine import _dedupe_opportunities

    ats = Opportunity(
        title="Staff Machine Learning Engineer - Edge AI",
        company="Samsara",
        url="https://www.samsara.com/company/careers/roles/7266357?gh_jid=7266357",
        pay_high=319_000,
        hours_per_week=40,
    )
    builtin = Opportunity(
        title="Staff Machine Learning Engineer - Samsara",
        company="Samsara",
        url="https://www.builtin.com/job/staff-machine-learning-engineer/7823375",
        pay_high=319_000,
        hours_per_week=40,
    )
    ads = Opportunity(
        title="Staff Machine Learning Engineer, Ads",
        company="Samsara",
        url="https://jobs.example/ads",
        pay_high=300_000,
        hours_per_week=40,
    )
    manager = Opportunity(
        title="Staff Machine Learning Engineer Manager",
        company="Samsara",
        url="https://jobs.example/mgr",
        pay_high=400_000,
        hours_per_week=40,
    )
    out = _dedupe_opportunities(
        sorted([builtin, ats, ads, manager], key=lambda o: o.score(), reverse=True)
    )
    urls = [o.url for o in out]
    assert urls[0] == manager.url
    assert ats.url in urls
    assert builtin.url not in urls
    assert ads.url in urls


def test_heuristic_company_from_lever_prefix():
    h = _heuristic_opportunity(
        {
            "title": "Lyra Health - Senior ML Engineer (ML/AI) - jobs.lever.co",
            "url": "https://jobs.lever.co/lyrahealth/d33ddfed-8c69-4e29-966b-0e190190cd6a",
            "description": "",
        }
    )
    assert h.company == "Lyra Health"


def test_heuristic_lever_requisition_suffix_is_not_company():
    h = _heuristic_opportunity(
        {
            "title": "IT Network Engineer II - 936",
            "url": "https://jobs.eu.lever.co/quantinuum/753dc869-e097-4ae9-89d1-81cf56de46a7",
            "description": "",
        }
    )
    assert h.company == "Quantinuum"


def test_heuristic_company_from_ashby_at():
    h = _heuristic_opportunity(
        {
            "title": "Senior ML Engineer @ Quilter",
            "url": "https://jobs.ashbyhq.com/quilter/2b0f95cb-7c8b-4b62-8bcb-b9993344f2f1/application",
            "description": "",
        }
    )
    assert h.company == "Quilter"
    assert h.url == "https://jobs.ashbyhq.com/quilter/2b0f95cb-7c8b-4b62-8bcb-b9993344f2f1"


def test_apply_listing_ashby_json_ld_pay():
    from src.engine import _apply_listing

    html = """
    <title>Senior ML Engineer @ Quilter</title>
    <script type="application/ld+json">
    {"@type":"JobPosting","hiringOrganization":{"name":"Quilter"},
     "baseSalary":{"currency":"USD","value":{"minValue":180000,"maxValue":200000,"unitText":"YEAR"}}}
    </script>
    """
    opp = Opportunity(
        title="Senior ML Engineer @ Quilter",
        url="https://jobs.ashbyhq.com/quilter/2b0f95cb-7c8b-4b62-8bcb-b9993344f2f1",
    )
    _apply_listing(opp, html)
    assert opp.company == "Quilter"
    assert opp.pay_low == 180_000
    assert opp.pay_high == 200_000


def test_heuristic_company_from_workable_apply_title():
    h = _heuristic_opportunity(
        {
            "title": "Senior Machine Learning Engineer - Multi Media LLC",
            "url": "https://apply.workable.com/multimediallc/j/73CB637EE8",
            "description": "",
        }
    )
    assert h.company == "Multi Media LLC"
    assert h.url == "https://apply.workable.com/multimediallc/j/73CB637EE8"


def test_heuristic_stores_workable_job_url_not_markdown():
    h = _heuristic_opportunity(
        {
            "title": "Senior Machine Learning Engineer",
            "url": "https://apply.workable.com/runware/jobs/view/B0A0A14125.md",
            "description": "",
        }
    )
    assert h.url == "https://apply.workable.com/runware/j/B0A0A14125"
    assert h.company == "Runware"


def test_apply_listing_workable_markdown_pay():
    from src.engine import _apply_listing, _workable_to_html

    md = """# Senior Machine Learning Engineer

> Multi Media LLC · United States (Remote) · Full-time · Posted 2026-06-01

**Salary:** USD 200,000–240,000

**Workplace:** remote
"""
    opp = Opportunity(
        title="Senior Machine Learning Engineer - Multi Media LLC",
        url="https://apply.workable.com/multimediallc/j/73CB637EE8",
    )
    _apply_listing(opp, _workable_to_html(md))
    assert opp.company == "Multi Media LLC"
    assert opp.pay_low == 200_000
    assert opp.pay_high == 240_000
    assert opp.hours_per_week == 40
    assert opp.remote is True


def test_workable_markdown_workplace_wins_over_body_hybrid():
    from src.engine import _apply_listing, _workable_to_html

    remote = Opportunity(
        title="x",
        url="https://apply.workable.com/canopy-7/j/D0F326A019",
        remote=True,
    )
    _apply_listing(
        remote,
        _workable_to_html(
            """# Senior Machine Learning Engineer

> Canopy · Detroit, United States (Remote) · Full-time

**Salary:** USD 126,000–180,000

**Workplace:** remote

Work in a hybrid environment and regularly commute to our Detroit office.
"""
        ),
    )
    assert remote.remote is True
    assert remote.pay_high == 180_000

    office = Opportunity(
        title="x",
        url="https://apply.workable.com/acme/j/AAAAAAAAAA",
        remote=True,
    )
    _apply_listing(
        office,
        _workable_to_html(
            """# Engineer

> Acme · San Francisco, CA · Full-time

**Salary:** USD 160,000–180,000

**Workplace:** on-site

Remote-friendly team. $160,000 - $180,000
"""
        ),
    )
    assert office.remote is False

    city = Opportunity(
        title="x",
        url="https://apply.workable.com/acme/j/BBBBBBBBBB",
        remote=True,
    )
    _apply_listing(
        city,
        _workable_to_html(
            """# Engineer

> Acme · San Francisco, CA · Full-time

**Salary:** USD 160,000–180,000
"""
        ),
    )
    assert city.remote is False


def test_listing_text_prefers_workable_markdown_over_spa_shell(monkeypatch):
    engine = Engine()
    seen: list[str] = []

    async def fake_get(_client, url: str) -> str:
        seen.append(url)
        if url.endswith(".md"):
            return (
                "# Senior Engineer, AI/ML\n\n"
                "> A2Z Sync · United States (Remote) · Full-time\n\n"
                "**Salary:** USD 160,000–190,000\n"
            )
        return "<title>Senior Engineer, AI/ML - A2Z Sync</title><p>Apply</p>"

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    html = asyncio.run(
        engine._listing_text("https://apply.workable.com/a2z-sync/j/C95E51CDDA")
    )
    assert seen[0] == "https://apply.workable.com/a2z-sync/jobs/view/C95E51CDDA.md"
    from src.engine import _apply_listing

    opp = Opportunity(
        title="Senior Engineer, AI/ML - A2Z Sync",
        url="https://apply.workable.com/a2z-sync/j/C95E51CDDA",
    )
    _apply_listing(opp, html)
    assert opp.company == "A2Z Sync"
    assert opp.pay_low == 160_000
    assert opp.pay_high == 190_000


def test_find_dedupes_workable_apply_and_jobs_board():
    engine = Engine()
    engine.openai = None

    async def fake_search(_query: str):
        return [
            {
                "title": "Senior Machine Learning Engineer - Multi Media LLC",
                "url": "https://apply.workable.com/multimediallc/j/73CB637EE8",
                "description": "USD 200,000–240,000",
            },
            {
                "title": "Senior Machine Learning Engineer | Multi Media LLC | Jobs By Workable",
                "url": "https://jobs.workable.com/view/bqkqSAJN2W35yHL1WmQ5C9/remote-machine-learning-engineer-in-united-states-at-multi-media-llc",
                "description": "",
            },
        ]

    engine._search_all = fake_search

    async def no_page(_url: str) -> str:
        return ""

    engine._listing_text = no_page
    ranked = asyncio.run(engine.find("ml", limit=20))
    assert [o.url for o in ranked] == [
        "https://apply.workable.com/multimediallc/j/73CB637EE8"
    ]
    assert ranked[0].pay_high == 240_000


def test_unify_workable_slug_with_real_name():
    from src.engine import _unify_board_companies

    named = Opportunity(
        title="Senior Machine Learning Engineer - Multi Media LLC",
        url="https://apply.workable.com/multimediallc/j/73CB637EE8",
        company="Multi Media LLC",
    )
    slugged = Opportunity(
        title="Other Role",
        url="https://apply.workable.com/multimediallc/j/AAAAAAAAAA",
        company="Multimediallc",
    )
    _unify_board_companies([named, slugged])
    assert slugged.company == "Multi Media LLC"


def test_heuristic_company_from_lever_slug():
    h = _heuristic_opportunity(
        {
            "title": "Senior ML Engineer (Portugal Based Remote/Hybrid)",
            "url": "https://jobs.lever.co/swordhealth/770e2ca0-a6a4-4ca9-9c0f-ce419284ddbe",
            "description": "",
        }
    )
    assert h.company == "Swordhealth"


def test_heuristic_title_company_wins_over_url_slug():
    h = _heuristic_opportunity(
        {
            "title": "Senior, ML Engineer - VLM at Torc Robotics",
            "url": "https://job-boards.greenhouse.io/torcrobotics/jobs/8572505002",
            "description": "",
        }
    )
    assert h.company == "Torc Robotics"


def test_heuristic_canonicalizes_greenhouse_embed_url():
    h = _heuristic_opportunity(
        {
            "title": "Senior Machine Learning Engineer",
            "url": "https://boards.greenhouse.io/embed/job_app?for=reddit&token=6960831",
            "description": "$216,700",
        }
    )
    assert h.url == "https://job-boards.greenhouse.io/reddit/jobs/6960831"
    assert h.company == "Reddit"


def test_find_dedupes_same_role_across_boards():
    engine = Engine()
    engine.openai = None

    async def fake_search(_query: str):
        return [
            {
                "title": "Senior ML Engineer (ML/AI) in Remote at Lyra Health",
                "url": "https://careers.example/lyra",
                "description": "$143,000 to 197,000",
            },
            {
                "title": "Lyra Health - Senior ML Engineer (ML/AI) - jobs.lever.co",
                "url": "https://jobs.lever.co/lyrahealth/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                "description": "$100k",
            },
            {
                "title": "Lyra Health - Sr. ML Engineer (MLOps)",
                "url": "https://jobs.lever.co/lyrahealth/ffffffff-1111-2222-3333-444444444444",
                "description": "$90k",
            },
        ]

    engine._search_all = fake_search

    async def no_page(_url: str) -> str:
        return ""

    engine._listing_text = no_page
    ranked = asyncio.run(engine.find("ml", limit=20))
    assert [o.url for o in ranked] == [
        "https://careers.example/lyra",
        "https://jobs.lever.co/lyrahealth/ffffffff-1111-2222-3333-444444444444",
    ]


def test_find_llm_grounds_urls_and_drops_hallucinations():
    engine = Engine()
    engine.openai = _fake_client(
        json.dumps(
            {
                "opportunities": [
                    {
                        "title": "Cheap LLM",
                        "company": "CoA",
                        "url": "https://jobs.example/a",
                        "pay_high": 100_000,
                        "hours_per_week": 40,
                        "remote": True,
                    },
                    {
                        "title": "Lush LLM",
                        "company": "CoB",
                        "url": "HTTPS://JOBS.EXAMPLE/B/",
                        "pay_high": 200_000,
                        "hours_per_week": 20,
                        "remote": True,
                    },
                    {
                        "title": "Hallucinated",
                        "url": "https://evil.example/nope",
                        "pay_high": 9_999_999,
                        "hours_per_week": 1,
                    },
                ]
            }
        )
    )

    async def fake_search(_query: str):
        return [
            {"title": "Raw A", "url": "https://jobs.example/a", "description": "$100k"},
            {"title": "Raw B", "url": "https://jobs.example/b", "description": "$200k, 20 hrs/week"},
        ]

    engine._search_all = fake_search
    ranked = asyncio.run(engine.find("contracts", limit=20))
    assert [o.title for o in ranked] == ["Lush LLM", "Cheap LLM"]
    assert ranked[0].url == "https://jobs.example/b"
    assert ranked[0].score() == 200.0


def test_extract_batch_prompt_asks_for_opportunities_object():
    captured: dict = {}
    engine = Engine()
    engine.openai = _fake_client('{"opportunities": []}', captured)
    batch = [{"title": "Senior Engineer", "url": "https://example.com/job", "description": "remote"}]
    asyncio.run(engine._extract_batch(batch, "ai engineer"))
    assert captured.get("response_format") == {"type": "json_object"}
    prompt = captured["messages"][0]["content"]
    assert "opportunities" in prompt
    assert "ai engineer" in prompt
    assert "Do not estimate pay or hours" in prompt


def test_extract_batch_keeps_raw_then_heuristics_when_llm_pay_hours_missing():
    engine = Engine()
    engine.openai = _fake_client(
        json.dumps(
            {
                "opportunities": [
                    {"title": "LLM title", "company": "LLM Co", "url": "https://keep-raw.example/a"},
                    {"title": "Needs guesses", "url": "https://guess.example/b"},
                ]
            }
        )
    )
    batch = [
        {
            "title": "Raw A",
            "url": "https://keep-raw.example/a",
            "description": "",
            "pay": 160_000,
            "hours": 20,
            "remote": True,
        },
        {
            "title": "Junior Developer",
            "url": "https://guess.example/b",
            "description": "hybrid office",
        },
    ]
    out = {o.url: o for o in asyncio.run(engine._extract_batch(batch, "q"))}
    kept = out["https://keep-raw.example/a"]
    assert kept.title == "LLM title"
    assert kept.pay_high == 160_000
    assert kept.hours_per_week == 20
    assert kept.efficiency == 160.0
    guessed = out["https://guess.example/b"]
    assert guessed.pay_high is None
    assert guessed.hours_per_week is None
    assert guessed.score() == 0
    assert guessed.remote is False


def test_extract_batch_falls_back_on_error_or_ungrounded_llm():
    boom = Engine()
    boom.openai = _fake_client_raises(RuntimeError("boom"))
    batch = [
        {
            "title": "Senior ML Engineer",
            "url": "https://fallback.example/1",
            "description": "contract",
        }
    ]
    failed = asyncio.run(boom._extract_batch(batch, "q"))
    assert failed[0].pay_high is None
    assert failed[0].hours_per_week is None
    assert failed[0].score() == 0
    assert failed[0].efficiency == failed[0].refined_rate

    ghost = Engine()
    ghost.openai = _fake_client(
        json.dumps(
            {
                "opportunities": [
                    {
                        "title": "Hallucinated",
                        "url": "https://not-in-batch.example/x",
                        "pay_high": 500_000,
                        "hours_per_week": 10,
                    }
                ]
            }
        )
    )
    grounded = asyncio.run(
        ghost._extract_batch(
            [
                {
                    "title": "Staff Engineer",
                    "url": "https://real.example/job",
                    "description": "fully remote",
                    "pay": 180_000,
                    "hours": 30,
                }
            ],
            "q",
        )
    )
    assert grounded[0].url == "https://real.example/job"
    assert grounded[0].title == "Staff Engineer"
    assert grounded[0].pay_high == 180_000


def test_extract_ignores_llm_invented_pay_and_hours():
    engine = Engine()
    engine.openai = _fake_client(
        json.dumps(
            {
                "opportunities": [
                    {
                        "title": "Inflated",
                        "url": "https://jobs.example/a",
                        "pay_high": 9_999_999,
                        "hours_per_week": 1,
                    }
                ]
            }
        )
    )
    out = asyncio.run(
        engine._extract_batch(
            [
                {
                    "title": "Engineer",
                    "url": "https://jobs.example/a",
                    "description": "no compensation listed",
                }
            ],
            "q",
        )
    )
    assert out[0].title == "Inflated"
    assert out[0].pay_high is None
    assert out[0].hours_per_week is None
    assert out[0].score() == 0


def test_find_ranks_parsed_pay_above_unknown():
    engine = Engine()
    engine.openai = None

    async def fake_search(_query: str):
        return [
            {
                "title": "Unknown pay",
                "url": "https://a.example/thin",
                "description": "Senior staff role",
            },
            {
                "title": "Priced",
                "url": "https://a.example/paid",
                "description": "$90k",
            },
        ]

    engine._search_all = fake_search

    async def no_page(_url: str) -> str:
        return ""

    engine._listing_text = no_page
    ranked = asyncio.run(engine.find("ml", limit=20))
    assert [o.title for o in ranked] == ["Priced", "Unknown pay"]
    assert ranked[0].score() == 45.0
    assert ranked[0].rate_is_imputed is True
    assert ranked[1].score() == 0


def test_heuristic_company_from_builtin_title():
    from src.engine import _apply_listing, _dedupe_opportunities, _role_title

    h = _heuristic_opportunity(
        {
            "title": "Enterprise Senior Lead AI Product Manager- Go to Market - Wells Fargo | Built In",
            "url": "https://www.builtin.com/job/enterprise-senior-lead-ai-product-manager-go-market/11015128",
            "description": "$185,000 - $300,000",
        }
    )
    assert h.company == "Wells Fargo"
    assert _role_title(h.title).endswith("Wells Fargo")
    assert "Built In" not in _role_title(h.title)
    html = (
        "<title>Substation Program Director - 26342 - Enverus | Built In</title>"
        "<p>In-Office. $130,000 - $150,000</p>"
    )
    opp = Opportunity(
        title="Substation Program Director - 26342 - Enverus | Built In",
        url="https://www.builtin.com/job/substation-program-director-26342/11015219",
    )
    _apply_listing(opp, html)
    assert opp.company == "Enverus"
    assert opp.pay_high == 150_000
    ats = Opportunity(
        title="Enterprise Senior Lead AI Product Manager- Go to Market",
        company="Wells Fargo",
        url="https://jobs.example/wf",
        pay_high=300_000,
    )
    builtin = Opportunity(
        title="Enterprise Senior Lead AI Product Manager- Go to Market - Wells Fargo | Built In",
        company="Wells Fargo",
        url="https://www.builtin.com/job/enterprise-senior-lead-ai-product-manager-go-market/11015128",
        pay_high=300_000,
    )
    assert [o.url for o in _dedupe_opportunities([ats, builtin])] == [ats.url]
    remote = _heuristic_opportunity(
        {
            "title": "ML Engineer - Remote | Built In",
            "url": "https://www.builtin.com/job/ml-engineer/1",
            "description": "",
        }
    )
    assert remote.company is None


def test_heuristic_company_from_wellfound_at_bullet_title():
    from src.engine import _dedupe_opportunities, _role_title

    h = _heuristic_opportunity(
        {
            "title": "IT Security Administrator at Bitwarden • Remote (Work from Home) | Wellfound",
            "url": "https://wellfound.com/jobs/4335648-it-security-administrator",
            "description": "$115,000 - $145,000",
        }
    )
    assert h.company == "Bitwarden"
    assert "Wellfound" not in _role_title(h.title)
    ats = Opportunity(
        title="IT Security Administrator",
        company="Bitwarden",
        url="https://jobs.lever.co/bitwarden/abc",
        pay_high=145_000,
    )
    wellfound = Opportunity(
        title="IT Security Administrator at Bitwarden • Remote (Work from Home) | Wellfound",
        company="Bitwarden",
        url="https://wellfound.com/jobs/4335648-it-security-administrator",
        pay_high=145_000,
    )
    assert [o.url for o in _dedupe_opportunities([ats, wellfound])] == [ats.url]
    v7 = _heuristic_opportunity(
        {
            "title": "Demand Generation Lead at V7 • New York City | Wellfound",
            "url": "https://wellfound.com/jobs/4677846-demand-generation-lead",
            "description": "",
        }
    )
    assert v7.company == "V7"


def test_heuristic_company_from_title_at():
    h = _heuristic_opportunity(
        {
            "title": "Senior Machine Learning Engineer at Lyra Health",
            "url": "https://job-boards.greenhouse.io/lyrahealth/jobs/123",
            "description": "Role in San Francisco.",
        }
    )
    assert h.company == "Lyra Health"


def test_heuristic_skips_at_remote():
    h = _heuristic_opportunity(
        {
            "title": "ML Engineer at Remote",
            "url": "https://example.com/jobs/1",
            "description": "Work from home.",
        }
    )
    assert h.company is None


def test_merge_company_from_raw_title_when_llm_omits_it():
    engine = Engine()
    engine.openai = _fake_client(
        json.dumps(
            {
                "opportunities": [
                    {
                        "title": "Senior Machine Learning Engineer",
                        "url": "https://job-boards.greenhouse.io/lyrahealth/jobs/123",
                    }
                ]
            }
        )
    )
    out = asyncio.run(
        engine._extract_batch(
            [
                {
                    "title": "Senior Machine Learning Engineer at Lyra Health",
                    "url": "https://job-boards.greenhouse.io/lyrahealth/jobs/123",
                    "description": "San Francisco",
                }
            ],
            "q",
        )
    )
    assert out[0].company == "Lyra Health"


def test_heuristic_range_and_imputed_hours():
    ranged = _heuristic_opportunity(
        {
            "title": "Eng $120k-$180k",
            "url": "https://example.com/range",
            "description": "",
        }
    )
    assert ranged.pay_low == 120_000
    assert ranged.pay_high == 180_000
    assert ranged.pay == 180_000
    assert ranged.hours_per_week is None
    assert ranged.rate_is_imputed is True
    assert ranged.refined_rate == 90.0


def test_enrich_pay_from_listing_html():
    engine = Engine()

    async def page(_url: str) -> str:
        return "for this full-time position is $143,000 to 197,000."

    engine._listing_text = page
    opp = Opportunity(title="Senior ML Engineer", url="https://careers.example/x")
    asyncio.run(engine._enrich_pay([opp]))
    assert opp.pay_low == 143_000
    assert opp.pay_high == 197_000
    assert opp.hours_per_week == 40
    assert opp.score() == 98.5


def test_apply_listing_json_ld_company_and_hourly_pay():
    from src.engine import _apply_listing

    html = """
    <script type="application/ld+json">
    {"@type":"JobPosting","title":"Senior ML Engineer",
     "hiringOrganization":{"@type":"Organization","name":"Braintrust"},
     "employmentType":"FULL_TIME",
     "baseSalary":{"@type":"MonetaryAmount","currency":"USD",
       "value":{"@type":"QuantitativeValue","minValue":80,"maxValue":100,"unitText":"HOUR"}}}
    </script>
    """
    opp = Opportunity(title="Senior ML Engineer", url="https://karkidi.example/x")
    _apply_listing(opp, html)
    assert opp.company == "Braintrust"
    assert opp.pay_low == 160_000
    assert opp.pay_high == 200_000
    assert opp.hours_per_week == 40
    assert opp.score() == 100.0


def test_apply_listing_prefers_html_yearly_over_json_ld_hourly():
    from src.engine import _apply_listing

    html = """
    <script type="application/ld+json">
    {"@type":"JobPosting","title":"Engineer",
     "baseSalary":{"@type":"MonetaryAmount","currency":"USD",
       "value":{"@type":"QuantitativeValue","value":80,"unitText":"HOUR"}}}
    </script>
    <p>Salary $180,000 plus $80/hr on-call</p>
    """
    opp = Opportunity(title="Engineer", url="https://jobs.example/ld-hr")
    assert _apply_listing(opp, html) is True
    assert opp.pay_high == 180_000
    day = """
    <script type="application/ld+json">
    {"@type":"JobPosting","title":"Engineer",
     "baseSalary":{"@type":"MonetaryAmount","currency":"USD",
       "value":{"@type":"QuantitativeValue","value":800,"unitText":"DAY"}}}
    </script>
    <p>Salary $180,000 plus $800/day travel</p>
    """
    travel = Opportunity(title="Engineer", url="https://jobs.example/ld-day")
    assert _apply_listing(travel, day) is True
    assert travel.pay_high == 180_000
    hourly = """
    <script type="application/ld+json">
    {"@type":"JobPosting","title":"Engineer",
     "baseSalary":{"@type":"MonetaryAmount","currency":"USD",
       "value":{"@type":"QuantitativeValue","minValue":80,"maxValue":100,"unitText":"HOUR"}}}
    </script>
    <p>Contract $80-$100/hr</p>
    """
    contract = Opportunity(title="Engineer", url="https://jobs.example/ld-only-hr")
    assert _apply_listing(contract, hourly) is True
    assert contract.pay_low == 160_000
    assert contract.pay_high == 200_000


def test_apply_listing_json_ld_yearly_thousands():
    from html import escape

    from src.engine import _apply_listing

    html = """
    <script type="application/ld+json">
    {"@type":"JobPosting","title":"Engineer",
     "baseSalary":{"@type":"MonetaryAmount","currency":"USD",
       "value":{"@type":"QuantitativeValue","minValue":150,"maxValue":180,"unitText":"YEAR"}}}
    </script>
    """
    opp = Opportunity(title="Engineer", url="https://jobs.example/ld-k")
    assert _apply_listing(opp, html) is True
    assert opp.pay_low == 150_000
    assert opp.pay_high == 180_000
    yearly = """
    <script type="application/ld+json">
    {"@type":"JobPosting","title":"Engineer",
     "baseSalary":{"@type":"MonetaryAmount","currency":"USD",
       "value":{"@type":"QuantitativeValue","value":180,"unitText":"YEARLY"}}}
    </script>
    """
    full = Opportunity(title="Engineer", url="https://jobs.example/ld-yearly")
    assert _apply_listing(full, yearly) is True
    assert full.pay_high == 180_000
    dollars = """
    <script type="application/ld+json">
    {"@type":"JobPosting","title":"Engineer",
     "baseSalary":{"@type":"MonetaryAmount","currency":"USD",
       "value":{"@type":"QuantitativeValue","minValue":150000,"maxValue":180000,"unitText":"YEAR"}}}
    </script>
    """
    exact = Opportunity(title="Engineer", url="https://jobs.example/ld-full")
    assert _apply_listing(exact, dollars) is True
    assert exact.pay_low == 150_000
    assert exact.pay_high == 180_000
    hourly = """
    <script type="application/ld+json">
    {"@type":"JobPosting","title":"Engineer",
     "baseSalary":{"@type":"MonetaryAmount","currency":"USD",
       "value":{"@type":"QuantitativeValue","minValue":80,"maxValue":100,"unitText":"HOUR"}}}
    </script>
    """
    hour = Opportunity(title="Engineer", url="https://jobs.example/ld-still-hr")
    assert _apply_listing(hour, hourly) is True
    assert hour.pay_low == 160_000
    assert hour.pay_high == 200_000
    estimated = """
    <script type="application/ld+json">
    {"@type":"JobPosting","title":"Engineer",
     "estimatedSalary":{"@type":"MonetaryAmount","currency":"USD",
       "value":{"@type":"QuantitativeValue","minValue":150000,"maxValue":180000,"unitText":"YEAR"}}}
    </script>
    """
    est = Opportunity(title="Engineer", url="https://jobs.example/ld-est")
    assert _apply_listing(est, estimated) is True
    assert est.pay_low == 150_000
    assert est.pay_high == 180_000
    listed = """
    <script type="application/ld+json">
    {"@type":"JobPosting","title":"Engineer",
     "baseSalary":{"@type":"MonetaryAmount","currency":"USD",
       "value":{"@type":"QuantitativeValue","value":180000,"unitText":"YEAR"}},
     "estimatedSalary":{"@type":"MonetaryAmount","currency":"USD",
       "value":{"@type":"QuantitativeValue","value":250000,"unitText":"YEAR"}}}
    </script>
    """
    base = Opportunity(title="Engineer", url="https://jobs.example/ld-base-est")
    assert _apply_listing(base, listed) is True
    assert base.pay_high == 180_000
    qv = """
    <script type="application/ld+json">
    {"@type":"JobPosting","title":"Engineer",
     "estimatedSalary":{"@type":"QuantitativeValue","minValue":150000,"maxValue":180000,"unitText":"YEAR"}}
    </script>
    """
    direct = Opportunity(title="Engineer", url="https://jobs.example/ld-est-qv")
    assert _apply_listing(direct, qv) is True
    assert direct.pay_low == 150_000
    assert direct.pay_high == 180_000
    day_est = """
    <script type="application/ld+json">
    {"@type":"JobPosting","title":"Engineer",
     "estimatedSalary":{"@type":"MonetaryAmount","currency":"USD",
       "value":{"@type":"QuantitativeValue","value":400,"unitText":"DAY"}}}
    </script>
    """
    diem = Opportunity(title="Engineer", url="https://jobs.example/ld-est-day")
    assert _apply_listing(diem, day_est) is True
    assert diem.pay_high == 100_000
    body = (
        '{"@type":"JobPosting","title":"Engineer",'
        '"hiringOrganization":{"name":"Acme"},'
        '"baseSalary":{"currency":"USD","value":{"minValue":180000,"maxValue":220000,"unitText":"YEAR"}}}'
    )
    charset = Opportunity(title="Engineer", url="https://jobs.example/ld-charset")
    assert _apply_listing(
        charset,
        f'<title>Engineer</title><script type="application/ld+json;charset=utf-8">{body}</script>',
    ) is True
    assert charset.pay_high == 220_000
    assert charset.company == "Acme"
    unquoted = Opportunity(title="Engineer", url="https://jobs.example/ld-unquoted")
    assert _apply_listing(
        unquoted,
        f"<title>Engineer</title><script type=application/ld+json>{body}</script>",
    ) is True
    assert unquoted.pay_high == 220_000
    cdata = Opportunity(title="Engineer", url="https://jobs.example/ld-cdata")
    assert _apply_listing(
        cdata,
        f'<title>Engineer</title><script type="application/ld+json">//<![CDATA[{body}//]]></script>',
    ) is True
    assert cdata.pay_high == 220_000
    wrapped = Opportunity(title="Engineer", url="https://jobs.example/ld-comment")
    assert _apply_listing(
        wrapped,
        f'<title>Engineer</title><script type="application/ld+json"><!--{body}--></script>',
    ) is True
    assert wrapped.pay_high == 220_000
    encoded = body.replace('"', "&quot;")
    entities = Opportunity(title="Engineer", url="https://jobs.example/ld-entities")
    assert _apply_listing(
        entities,
        f'<title>Engineer</title><script type="application/ld+json">{encoded}</script>',
    ) is True
    assert entities.pay_high == 220_000
    assert entities.company == "Acme"
    amp = body.replace("Acme", "Acme & Co")
    escaped = Opportunity(title="Engineer", url="https://jobs.example/ld-escape")
    assert _apply_listing(
        escaped,
        f'<title>Engineer</title><script type="application/ld+json">{escape(amp)}</script>',
    ) is True
    assert escaped.pay_high == 220_000
    assert escaped.company == "Acme & Co"
    numeric = Opportunity(title="Engineer", url="https://jobs.example/ld-numquot")
    assert _apply_listing(
        numeric,
        f'<title>Engineer</title><script type="application/ld+json">{body.replace(chr(34), "&#34;")}</script>',
    ) is True
    assert numeric.pay_high == 220_000
    inner = (
        '{"@type":"JobPosting","title":"Engineer",'
        '"description":"Use &quot;quotes&quot; in HTML",'
        '"hiringOrganization":{"name":"Acme"},'
        '"baseSalary":{"currency":"USD","value":{"minValue":180000,"maxValue":220000,"unitText":"YEAR"}}}'
    )
    quoted = Opportunity(title="Engineer", url="https://jobs.example/ld-inner-quot")
    assert _apply_listing(
        quoted,
        f'<title>Engineer</title><script type="application/ld+json">{inner}</script>',
    ) is True
    assert quoted.pay_high == 220_000
    bom = Opportunity(title="Engineer", url="https://jobs.example/ld-bom")
    assert _apply_listing(
        bom,
        f'<title>Engineer</title><script type="application/ld+json">\ufeff{body}</script>',
    ) is True
    assert bom.pay_high == 220_000


def test_apply_listing_json_ld_monthly_and_weekly_thousands():
    from src.engine import _apply_listing

    html = """
    <script type="application/ld+json">
    {"@type":"JobPosting","title":"Engineer",
     "baseSalary":{"@type":"MonetaryAmount","currency":"USD",
       "value":{"@type":"QuantitativeValue","minValue":15,"maxValue":18,"unitText":"MONTH"}}}
    </script>
    """
    opp = Opportunity(title="Engineer", url="https://jobs.example/ld-mo-k")
    assert _apply_listing(opp, html) is True
    assert opp.pay_low == 180_000
    assert opp.pay_high == 216_000
    full = """
    <script type="application/ld+json">
    {"@type":"JobPosting","title":"Engineer",
     "baseSalary":{"@type":"MonetaryAmount","currency":"USD",
       "value":{"@type":"QuantitativeValue","value":15000,"unitText":"MONTH"}}}
    </script>
    """
    dollars = Opportunity(title="Engineer", url="https://jobs.example/ld-mo")
    assert _apply_listing(dollars, full) is True
    assert dollars.pay_high == 180_000
    week = """
    <script type="application/ld+json">
    {"@type":"JobPosting","title":"Engineer",
     "baseSalary":{"@type":"MonetaryAmount","currency":"USD",
       "value":{"@type":"QuantitativeValue","value":3,"unitText":"WEEK"}}}
    </script>
    """
    wk = Opportunity(title="Engineer", url="https://jobs.example/ld-wk-k")
    assert _apply_listing(wk, week) is True
    assert wk.pay_high == 150_000
    week_full = """
    <script type="application/ld+json">
    {"@type":"JobPosting","title":"Engineer",
     "baseSalary":{"@type":"MonetaryAmount","currency":"USD",
       "value":{"@type":"QuantitativeValue","value":3000,"unitText":"WEEK"}}}
    </script>
    """
    wk_d = Opportunity(title="Engineer", url="https://jobs.example/ld-wk")
    assert _apply_listing(wk_d, week_full) is True
    assert wk_d.pay_high == 150_000
    biweek = """
    <script type="application/ld+json">
    {"@type":"JobPosting","title":"Engineer",
     "baseSalary":{"@type":"MonetaryAmount","currency":"USD",
       "value":{"@type":"QuantitativeValue","value":3,"unitText":"BIWEEKLY"}}}
    </script>
    """
    bi = Opportunity(title="Engineer", url="https://jobs.example/ld-bi-thou")
    assert _apply_listing(bi, biweek) is True
    assert bi.pay_high == 75_000
    bi_full = """
    <script type="application/ld+json">
    {"@type":"JobPosting","title":"Engineer",
     "baseSalary":{"@type":"MonetaryAmount","currency":"USD",
       "value":{"@type":"QuantitativeValue","value":3000,"unitText":"BIWEEKLY"}}}
    </script>
    """
    bi_d = Opportunity(title="Engineer", url="https://jobs.example/ld-bi-full")
    assert _apply_listing(bi_d, bi_full) is True
    assert bi_d.pay_high == 75_000
    semi = """
    <script type="application/ld+json">
    {"@type":"JobPosting","title":"Engineer",
     "baseSalary":{"@type":"MonetaryAmount","currency":"USD",
       "value":{"@type":"QuantitativeValue","value":3,"unitText":"SEMIMONTHLY"}}}
    </script>
    """
    sm = Opportunity(title="Engineer", url="https://jobs.example/ld-sm-thou")
    assert _apply_listing(sm, semi) is True
    assert sm.pay_high == 72_000


def test_apply_listing_empty_json_ld_salary_falls_back_to_visible_text():
    from src.engine import _apply_listing

    html = """
    <script type="application/ld+json">
    {"@type":"JobPosting","hiringOrganization":{"name":"Lyra Health"},
     "baseSalary":{"@type":"MonetaryAmount","currency":"","value":{"unitText":""}}}
    </script>
    <p>for this full-time position is $143,000 to 197,000.</p>
    """
    opp = Opportunity(title="Senior ML Engineer", url="https://careers.example/x")
    _apply_listing(opp, html)
    assert opp.company == "Lyra Health"
    assert opp.pay_low == 143_000
    assert opp.pay_high == 197_000


def test_apply_listing_ignores_non_usd_salary():
    from src.engine import _apply_listing

    html = """
    <script type="application/ld+json">
    {"@type":"JobPosting","hiringOrganization":{"name":"Acme"},
     "baseSalary":{"currency":"EUR","value":{"minValue":120000,"maxValue":180000,"unitText":"YEAR"}}}
    </script>
    """
    opp = Opportunity(title="Engineer", url="https://jobs.example/x")
    _apply_listing(opp, html)
    assert opp.company == "Acme"
    assert opp.pay_high is None
    estimated = """
    <script type="application/ld+json">
    {"@type":"JobPosting","hiringOrganization":{"name":"Acme"},
     "estimatedSalary":{"currency":"EUR","value":{"minValue":120000,"maxValue":180000,"unitText":"YEAR"}}}
    </script>
    <p>Account Executive $220,000</p>
    """
    est = Opportunity(title="Engineer", url="https://jobs.example/est-eur")
    _apply_listing(est, estimated)
    assert est.pay_high is None


def test_apply_listing_json_ld_amount_without_currency_follows_country():
    from src.engine import _apply_listing, _foreign_salary

    se = """
    <script type="application/ld+json">
    {"@type":"JobPosting","hiringOrganization":{"name":"Spotify"},
     "jobLocation":{"address":{"addressCountry":"SE"}},
     "baseSalary":{"currency":"","value":{"minValue":800000,"maxValue":900000,"unitText":"YEAR"}}}
    </script>
    <p>US equivalent $90,000 a year</p>
    """
    se_opp = Opportunity(title="Engineer", url="https://jobs.example/se")
    assert _apply_listing(se_opp, se) is False
    assert se_opp.company == "Spotify"
    assert se_opp.pay_high is None
    assert _foreign_salary(se) is True

    dict_se = """
    <script type="application/ld+json">
    {"@type":"JobPosting","hiringOrganization":{"name":"Volvo"},
     "jobLocation":{"address":{"addressCountry":{"@type":"Country","name":"Sweden"}}},
     "baseSalary":800000}
    </script>
    """
    dict_opp = Opportunity(title="Engineer", url="https://jobs.example/volvo")
    assert _apply_listing(dict_opp, dict_se) is False
    assert dict_opp.pay_high is None
    assert _foreign_salary(dict_se) is True

    us = """
    <script type="application/ld+json">
    {"@type":"JobPosting","hiringOrganization":{"name":"Acme"},
     "jobLocation":{"address":{"addressCountry":"United States"}},
     "baseSalary":{"currency":"","value":{"minValue":180000,"maxValue":200000,"unitText":"YEAR"}}}
    </script>
    """
    us_opp = Opportunity(title="Engineer", url="https://jobs.example/us")
    assert _apply_listing(us_opp, us) is True
    assert us_opp.pay_low == 180_000
    assert us_opp.pay_high == 200_000
    assert _foreign_salary(us) is False

    none = """
    <script type="application/ld+json">
    {"@type":"JobPosting","hiringOrganization":{"name":"Acme"},
     "baseSalary":{"value":{"minValue":180000,"maxValue":200000,"unitText":"YEAR"}}}
    </script>
    """
    none_opp = Opportunity(title="Engineer", url="https://jobs.example/none")
    assert _apply_listing(none_opp, none) is True
    assert none_opp.pay_high == 200_000
    assert _foreign_salary(none) is False

    usd_se = """
    <script type="application/ld+json">
    {"@type":"JobPosting","hiringOrganization":{"name":"Acme"},
     "jobLocation":{"address":{"addressCountry":"SE"}},
     "baseSalary":{"currency":"USD","value":{"minValue":180000,"maxValue":200000,"unitText":"YEAR"}}}
    </script>
    """
    usd_opp = Opportunity(title="Engineer", url="https://jobs.example/usd-se")
    assert _apply_listing(usd_opp, usd_se) is True
    assert usd_opp.pay_high == 200_000
    assert _foreign_salary(usd_se) is False


def test_apply_listing_json_ld_amount_without_currency_reads_place_name():
    from src.engine import _apply_listing, _foreign_salary

    se = """
    <script type="application/ld+json">
    {"@type":"JobPosting","hiringOrganization":{"name":"Spotify"},
     "jobLocation":{"@type":"Place","name":"Stockholm, Sweden"},
     "baseSalary":{"value":{"minValue":800000,"maxValue":900000,"unitText":"YEAR"}}}
    </script>
    <p>US equivalent $90,000 a year</p>
    """
    se_opp = Opportunity(title="Engineer", url="https://jobs.example/sthlm")
    assert _apply_listing(se_opp, se) is False
    assert se_opp.pay_high is None
    assert _foreign_salary(se) is True

    uk = """
    <script type="application/ld+json">
    {"@type":"JobPosting","hiringOrganization":{"name":"Acme"},
     "jobLocation":"London, UK",
     "baseSalary":{"value":{"minValue":80000,"maxValue":100000,"unitText":"YEAR"}}}
    </script>
    """
    uk_opp = Opportunity(title="Engineer", url="https://jobs.example/uk")
    assert _apply_listing(uk_opp, uk) is False
    assert uk_opp.pay_high is None
    assert _foreign_salary(uk) is True

    us = """
    <script type="application/ld+json">
    {"@type":"JobPosting","hiringOrganization":{"name":"Acme"},
     "jobLocation":{"@type":"Place","name":"Austin, TX, United States"},
     "baseSalary":{"value":{"minValue":180000,"maxValue":200000,"unitText":"YEAR"}}}
    </script>
    """
    us_opp = Opportunity(title="Engineer", url="https://jobs.example/austin")
    assert _apply_listing(us_opp, us) is True
    assert us_opp.pay_high == 200_000
    assert _foreign_salary(us) is False

    empty = """
    <script type="application/ld+json">
    {"@type":"JobPosting","hiringOrganization":{"name":"Klarna"},
     "jobLocation":{"@type":"Place","name":"Stockholm, Sweden"},
     "baseSalary":{"currency":"","value":{"unitText":""}}}
    </script>
    <p>$180,000 a year</p>
    """
    empty_opp = Opportunity(title="Engineer", url="https://jobs.example/empty-sthlm")
    assert _apply_listing(empty_opp, empty) is True
    assert empty_opp.pay_high == 180_000
    assert _foreign_salary(empty) is False


def test_apply_listing_empty_json_ld_salary_non_us_falls_back_to_visible_text():
    from src.engine import _apply_listing, _foreign_salary

    html = """
    <script type="application/ld+json">
    {"@type":"JobPosting","hiringOrganization":{"name":"Klarna"},
     "jobLocation":{"address":{"addressCountry":"Sweden"}},
     "baseSalary":{"@type":"MonetaryAmount","currency":"","value":{"unitText":""}}}
    </script>
    <p>$180,000 a year</p>
    """
    opp = Opportunity(title="Engineer", url="https://jobs.example/klarna")
    assert _apply_listing(opp, html) is True
    assert opp.company == "Klarna"
    assert opp.pay_high == 180_000
    assert _foreign_salary(html) is False


def test_enrich_drops_foreign_salary_keeps_unknown_usd():
    engine = Engine()

    async def page(url: str) -> str:
        if "eur" in url:
            return """
            <title>Senior ML Engineer</title>
            <p>€60,000 - €85,000 a year</p>
            """
        if "jsonld" in url:
            return """
            <script type="application/ld+json">
            {"@type":"JobPosting","hiringOrganization":{"name":"Acme"},
             "baseSalary":{"currency":"EUR","value":{"minValue":120000,"maxValue":180000,"unitText":"YEAR"}}}
            </script>
            """
        if "usd" in url:
            return "<title>Senior ML</title><p>$180,000 a year</p>"
        return "<title>Staff Engineer</title><p>Apply now. No salary listed.</p>"

    engine._listing_text = page
    usd = Opportunity(title="USD", url="https://jobs.example/usd")
    eur = Opportunity(title="EUR", url="https://jobs.example/eur")
    jsonld = Opportunity(title="JSON", url="https://jobs.example/jsonld")
    unknown = Opportunity(title="Unknown", url="https://jobs.example/unknown")
    opps = [usd, eur, jsonld, unknown]
    asyncio.run(engine._enrich_pay(opps))
    assert [o.title for o in opps] == ["USD", "Unknown"]
    assert usd.pay_high == 180_000
    assert unknown.pay_high is None


def test_enrich_drops_json_ld_amount_without_currency_outside_us():
    engine = Engine()

    async def page(url: str) -> str:
        if "se" in url:
            return """
            <script type="application/ld+json">
            {"@type":"JobPosting","hiringOrganization":{"name":"Spotify"},
             "jobLocation":{"address":{"addressCountry":"SE"}},
             "baseSalary":{"value":{"minValue":800000,"maxValue":900000,"unitText":"YEAR"}}}
            </script>
            """
        return "<title>Staff Engineer</title><p>Apply now. No salary listed.</p>"

    engine._listing_text = page
    se = Opportunity(title="SE", url="https://jobs.example/se")
    unknown = Opportunity(title="Unknown", url="https://jobs.example/unknown")
    opps = [se, unknown]
    asyncio.run(engine._enrich_pay(opps))
    assert [o.title for o in opps] == ["Unknown"]
    assert unknown.pay_high is None


def test_enrich_drops_foreign_listing_even_when_snippet_has_dollars():
    engine = Engine()

    async def page(url: str) -> str:
        if "eur" in url:
            return """
            <title>Senior ML Engineer</title>
            <p>€60,000 - €85,000 a year</p>
            """
        if "jsonld" in url:
            return """
            <script type="application/ld+json">
            {"@type":"JobPosting","hiringOrganization":{"name":"Acme"},
             "baseSalary":{"currency":"EUR","value":{"minValue":120000,"maxValue":180000,"unitText":"YEAR"}}}
            </script>
            """
        return "<title>Staff Engineer</title><p>Apply now. No salary listed.</p>"

    engine._listing_text = page
    eur = Opportunity(
        title="EUR",
        url="https://jobs.example/eur",
        company="Acme",
        pay_high=180_000,
        hours_per_week=40,
    )
    jsonld = Opportunity(
        title="JSON",
        url="https://jobs.example/jsonld",
        company="Acme",
        pay_high=180_000,
        hours_per_week=40,
    )
    unknown = Opportunity(
        title="Unknown",
        url="https://jobs.example/unknown",
        company="Acme",
        pay_high=90_000,
    )
    opps = [eur, jsonld, unknown]
    asyncio.run(engine._enrich_pay(opps))
    assert [o.title for o in opps] == ["Unknown"]
    assert unknown.pay_high == 90_000


def test_enrich_drops_foreign_k_suffix_pay():
    engine = Engine()

    async def page(url: str) -> str:
        if "gbp" in url:
            return "<title>Engineer</title><p>£60K - £80K plus equity</p>"
        return "<title>Engineer</title><p>Apply now. No salary listed.</p>"

    engine._listing_text = page
    gbp = Opportunity(title="GBP", url="https://jobs.example/gbp", company="Acme")
    unknown = Opportunity(title="Unknown", url="https://jobs.example/unknown", company="Acme")
    opps = [gbp, unknown]
    asyncio.run(engine._enrich_pay(opps))
    assert [o.title for o in opps] == ["Unknown"]
    assert unknown.pay_high is None


def test_enrich_drops_chf_listing_even_when_usd_equivalent_is_stated():
    engine = Engine()

    async def page(url: str) -> str:
        if "chf" in url:
            return (
                "<title>ML Engineer</title>"
                "<p>The salary is CHF 150,000. US equivalent $180,000</p>"
            )
        return "<title>Engineer</title><p>Apply now. No salary listed.</p>"

    engine._listing_text = page
    chf = Opportunity(
        title="CHF",
        url="https://jobs.example/chf",
        company="Acme",
        pay_high=180_000,
        hours_per_week=40,
    )
    unknown = Opportunity(title="Unknown", url="https://jobs.example/unknown", company="Acme")
    opps = [chf, unknown]
    asyncio.run(engine._enrich_pay(opps))
    assert [o.title for o in opps] == ["Unknown"]


def test_enrich_drops_salario_dollar_pay_even_when_snippet_has_dollars():
    engine = Engine()

    async def page(url: str) -> str:
        if "mx" in url:
            return (
                "<title>Account Manager Lead</title>"
                "<p>Salario bruto mensual entre $20,000 y $25,000</p>"
            )
        return "<title>Engineer</title><p>Apply now. No salary listed.</p>"

    engine._listing_text = page
    mx = Opportunity(
        title="MX",
        url="https://jobs.example/mx",
        company="Lyra",
        pay_high=180_000,
        hours_per_week=40,
    )
    unknown = Opportunity(title="Unknown", url="https://jobs.example/unknown", company="Lyra")
    opps = [mx, unknown]
    asyncio.run(engine._enrich_pay(opps))
    assert [o.title for o in opps] == ["Unknown"]


def test_enrich_fetches_when_company_missing_even_if_paid():
    engine = Engine()
    captured: list[str] = []

    async def page(url: str) -> str:
        captured.append(url)
        return """
        <script type="application/ld+json">
        {"@type":"JobPosting","hiringOrganization":{"name":"Braintrust"}}
        </script>
        """

    engine._listing_text = page
    opp = Opportunity(
        title="Senior ML Engineer",
        url="https://karkidi.example/x",
        pay_low=160_000,
        pay_high=200_000,
    )
    asyncio.run(engine._enrich_pay([opp]))
    assert captured == ["https://karkidi.example/x"]
    assert opp.company == "Braintrust"
    assert opp.pay_high == 200_000


def test_enrich_fetches_paid_named_listings_for_hours_and_gone_jobs():
    engine = Engine()

    async def page(url: str):
        if "gone" in url:
            return None
        if "thin" in url:
            return ""
        return """
        <script type="application/ld+json">
        {"@type":"JobPosting","title":"Senior ML Engineer",
         "hiringOrganization":{"name":"Quilter"},
         "employmentType":"FULL_TIME",
         "baseSalary":{"currency":"USD","value":{"minValue":180000,"maxValue":200000,"unitText":"YEAR"}}}
        </script>
        """

    engine._listing_text = page
    priced = Opportunity(
        title="Senior ML Engineer @ Quilter",
        url="https://jobs.ashbyhq.com/quilter/live",
        company="Quilter",
        pay_high=100_000,
        hours_per_week=None,
    )
    ghost = Opportunity(
        title="Expired",
        url="https://jobs.ashbyhq.com/azx/gone",
        company="AZX",
        pay_high=140_000,
        hours_per_week=40,
    )
    thin = Opportunity(
        title="Timeout",
        url="https://jobs.ashbyhq.com/weave/thin",
        company="Weave",
        pay_high=90_000,
    )
    opps = [priced, ghost, thin]
    asyncio.run(engine._enrich_pay(opps))
    assert [o.url for o in opps] == [
        "https://jobs.ashbyhq.com/quilter/live",
        "https://jobs.ashbyhq.com/weave/thin",
    ]
    assert priced.pay_low == 180_000
    assert priced.pay_high == 200_000
    assert priced.hours_per_week == 40
    assert priced.rate_is_imputed is False
    assert thin.pay_high == 90_000
    assert thin.hours_per_week is None


def test_unify_board_companies_prefers_real_name_over_slug():
    from src.engine import _unify_board_companies

    named = Opportunity(
        title="Sword Health - Senior ML Engineer (Europe-based/Remote)",
        url="https://jobs.lever.co/swordhealth/50945411-2f43-421a-8bb8-86aa1de6d890",
        company="Sword Health",
    )
    slugged = Opportunity(
        title="Senior ML Engineer (Portugal Based Remote/Hybrid)",
        url="https://jobs.lever.co/swordhealth/770e2ca0-a6a4-4ca9-9c0f-ce419284ddbe",
        company="Swordhealth",
    )
    other = Opportunity(
        title="Egen - Senior AI Engineer",
        url="https://jobs.lever.co/egen/1b870652-5768-45e9-b55b-4420e6402314",
        company="Egen",
    )
    _unify_board_companies([named, slugged, other])
    assert slugged.company == "Sword Health"
    assert named.company == "Sword Health"
    assert other.company == "Egen"


def test_enrich_unifies_slug_company_when_listings_already_priced():
    engine = Engine()

    async def page(_url: str) -> str:
        return ""

    engine._listing_text = page
    named = Opportunity(
        title="Sword Health - Senior ML",
        url="https://jobs.lever.co/swordhealth/aaa",
        company="Sword Health",
        pay_high=100_000,
    )
    slugged = Opportunity(
        title="Senior ML Engineer (Portugal)",
        url="https://jobs.lever.co/swordhealth/bbb",
        company="Swordhealth",
        pay_high=100_000,
    )
    asyncio.run(engine._enrich_pay([named, slugged]))
    assert slugged.company == "Sword Health"


def test_apply_listing_reads_json_ld_past_first_80k():
    from src.engine import _apply_listing

    html = (
        "<html><head><title>Role</title></head><body>"
        + ("x" * 81_000)
        + """<script type="application/ld+json">
        {"@type":"JobPosting","hiringOrganization":{"name":"Sword Health"}}
        </script></body></html>"""
    )
    opp = Opportunity(title="Senior ML Engineer", url="https://jobs.lever.co/swordhealth/1")
    _apply_listing(opp, html)
    assert opp.company == "Sword Health"


def test_apply_listing_pay_not_blocked_by_css_dollar_prefix():
    from src.engine import _apply_listing

    html = (
        "<style>" + ("$iconThumbnailMarginX;" * 5000) + "</style>"
        "<p>for this full-time position is $143,000 to 197,000.</p>"
    )
    opp = Opportunity(title="Senior ML Engineer", url="https://jobs.lever.co/lyrahealth/x")
    _apply_listing(opp, html)
    assert opp.pay_low == 143_000
    assert opp.pay_high == 197_000


def test_listing_text_fetches_lever_job_not_apply_form(monkeypatch):
    engine = Engine()
    seen: list[str] = []

    async def fake_get(_client, url: str) -> str:
        seen.append(url)
        if "api.lever.co" in url:
            return json.dumps(
                {
                    "id": "0bf1decc-002c-4b0a-b97b-6407d2930fff",
                    "text": "Senior AI/ML Engineer (GenAI, AWS)",
                    "categories": {"commitment": "Full-time"},
                    "salaryRange": {
                        "min": 159300,
                        "max": 219245,
                        "currency": "USD",
                        "interval": "per-year-salary",
                    },
                    "description": "<p>Build GenAI systems.</p>",
                }
            )
        return "<title>Provectus - Senior ML</title>"

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    html = asyncio.run(
        engine._listing_text(
            "https://jobs.lever.co/provectus/0bf1decc-002c-4b0a-b97b-6407d2930fff/apply"
        )
    )
    assert seen == [
        "https://api.lever.co/v0/postings/provectus/0bf1decc-002c-4b0a-b97b-6407d2930fff"
    ]
    from src.engine import _apply_listing

    opp = Opportunity(title="x", url="https://jobs.lever.co/provectus/0bf1decc-002c-4b0a-b97b-6407d2930fff")
    _apply_listing(opp, html)
    assert opp.title == "Senior AI/ML Engineer (GenAI, AWS)"
    assert opp.pay_low == 159_300
    assert opp.pay_high == 219_245
    assert opp.hours_per_week == 40


def test_lever_api_url_uses_eu_host():
    from src.engine import _lever_api_url

    assert _lever_api_url(
        "https://jobs.lever.co/provectus/0bf1decc-002c-4b0a-b97b-6407d2930fff/apply"
    ) == "https://api.lever.co/v0/postings/provectus/0bf1decc-002c-4b0a-b97b-6407d2930fff"
    assert _lever_api_url(
        "https://jobs.eu.lever.co/prima/cc0b6018-ef61-453f-8201-ab5e6db53e31"
    ) == "https://api.eu.lever.co/v0/postings/prima/cc0b6018-ef61-453f-8201-ab5e6db53e31"
    assert _lever_api_url(
        "https://jobs.eu.lever.co/quantinuum/753dc869-e097-4ae9-89d1-81cf56de46a7/apply"
    ) == "https://api.eu.lever.co/v0/postings/quantinuum/753dc869-e097-4ae9-89d1-81cf56de46a7"
    assert _lever_api_url("https://jobs.eu.lever.co/prima") is None


def test_lever_company_board_is_not_a_job():
    from src.engine import _heuristic_opportunity, _is_index_page, _lever_is_board

    board = "https://jobs.eu.lever.co/tomtom"
    posting = (
        "https://jobs.eu.lever.co/tomtom/ca7ff74f-4ebb-4e10-91ca-096d7faa89b7"
    )
    assert _lever_is_board(board) is True
    assert _lever_is_board("https://jobs.lever.co/spotify") is True
    assert _lever_is_board(posting) is False
    assert _lever_is_board(posting + "/apply") is False
    assert _is_index_page(
        {"url": board, "title": "TomTom - Lever", "description": ""}
    )
    assert not _is_index_page(
        {
            "url": posting,
            "title": "Machine Learning Staff Engineer – ADAS Online",
            "description": "",
        }
    )
    assert (
        _heuristic_opportunity(
            {"url": board, "title": "TomTom - Lever", "description": ""}
        )
        is None
    )


def test_listing_text_lever_board_is_gone(monkeypatch):
    engine = Engine()

    async def fake_get(_client, _url: str):
        raise AssertionError("Lever board HTML must not be fetched")

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    assert asyncio.run(engine._listing_text("https://jobs.eu.lever.co/tomtom")) is None
    assert asyncio.run(engine._listing_text("https://jobs.lever.co/spotify")) is None


def test_listing_text_greenhouse_and_ashby_boards_are_gone(monkeypatch):
    engine = Engine()

    async def fake_get(_client, _url: str):
        raise AssertionError("ATS board HTML must not be fetched")

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    assert (
        asyncio.run(engine._listing_text("https://job-boards.greenhouse.io/reddit"))
        is None
    )
    assert asyncio.run(engine._listing_text("https://boards.greenhouse.io/figma")) is None
    assert (
        asyncio.run(engine._listing_text("https://www.greenhouse.com/careers")) is None
    )
    assert asyncio.run(engine._listing_text("https://jobs.ashbyhq.com/webai")) is None


def test_listing_text_reads_lever_eu_api(monkeypatch):
    engine = Engine()
    seen: list[str] = []

    async def fake_get(_client, url: str) -> str:
        seen.append(url)
        if "api.eu.lever.co" in url:
            return json.dumps(
                {
                    "id": "753dc869-e097-4ae9-89d1-81cf56de46a7",
                    "text": "IT Network Engineer II",
                    "workplaceType": "remote",
                    "categories": {"commitment": "Full-time"},
                    "salaryRange": {
                        "min": 86000,
                        "max": 108000,
                        "currency": "USD",
                        "interval": "per-year-salary",
                    },
                    "description": "<p>Run the network.</p>",
                }
            )
        return "<title>Jobs at Quantinuum</title>"

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    html = asyncio.run(
        engine._listing_text(
            "https://jobs.eu.lever.co/quantinuum/753dc869-e097-4ae9-89d1-81cf56de46a7"
        )
    )
    assert seen == [
        "https://api.eu.lever.co/v0/postings/quantinuum/753dc869-e097-4ae9-89d1-81cf56de46a7"
    ]
    from src.engine import _apply_listing

    opp = Opportunity(
        title="x",
        url="https://jobs.eu.lever.co/quantinuum/753dc869-e097-4ae9-89d1-81cf56de46a7",
    )
    _apply_listing(opp, html)
    assert opp.title == "IT Network Engineer II"
    assert opp.pay_low == 86_000
    assert opp.pay_high == 108_000
    assert opp.remote is True
    assert opp.score() == 54.0
    assert opp.company == "Quantinuum"


def test_apply_listing_company_from_html_title():
    from src.engine import _apply_listing

    html = "<title>Job Application for Senior ML Engineer I // II at Signifyd</title><p>Apply</p>"
    opp = Opportunity(title="Senior Machine Learning Engineer I // II", url="https://job-boards.greenhouse.io/signifyd95/jobs/1")
    _apply_listing(opp, html)
    assert opp.company == "Signifyd"


def test_html_is_gone_removed_listing_banner():
    from src.engine import _html_is_gone, _html_is_index

    removed = (
        "<title>Machine learning Engineer - Agentic Retrieval - Zoom | Built In</title>"
        "<p>Sorry, this job was removed at 08:39 a.m. (UTC) on Friday, Aug 28, 2026</p>"
        "<p>Similar Jobs Square Account Executive $151,800 - $332,200</p>"
    )
    assert _html_is_gone(removed) is True
    assert _html_is_index(removed, "https://www.builtin.com/job/ml/10377512") is False
    assert (
        _html_is_gone(
            "<title>ML Engineer - Zoom | Built In</title>"
            "<p>Design hybrid retrieval systems. $180,000 - $220,000</p>"
        )
        is False
    )
    assert (
        _html_is_gone(
            "<title>Engineer</title>"
            "<p>Applications removed from consideration stay on file.</p>"
        )
        is False
    )
    filled = (
        "<title>Senior ML Engineer - Acme</title>"
        "<p>This position has been filled.</p>"
        "<p>$180,000 - $220,000 a year</p>"
    )
    assert _html_is_gone(filled) is True
    assert (
        _html_is_gone(
            "<title>Engineer</title>"
            "<p>Once this position has been filled, the team will grow.</p>"
            "<p>$180,000 a year</p>"
        )
        is False
    )
    assert _html_is_gone(
        "<title>Engineer</title><p>This job posting has expired.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title><p>This posting is no longer active.</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title><p>We're no longer hiring for this role.</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title><p>The position has been filled.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title><p>This vacancy has been filled.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title><p>This opportunity has been filled.</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title><p>We have filled this position.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title><p>This job is closed.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title><p>This posting has been closed.</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title><p>This opportunity is no longer available.</p>"
        "<p>$180,000 a year</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title><p>This job is no longer posted.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title><p>This requisition is closed.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title><p>This vacancy is closed.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title><p>We have closed this position.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title><p>We are no longer accepting applications.</p>"
        "<p>$180,000 a year</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title><p>This listing is no longer available.</p>"
        "<p>$180,000 a year</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title><p>This opening has been filled.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title><p>This job has been withdrawn.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title><p>This posting was withdrawn.</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title><p>This requisition has been cancelled.</p>"
        "<p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title><p>This listing has been taken down.</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title><p>This job is no longer listed.</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>Once the position has been filled, the team will grow.</p>"
        "<p>$180,000 a year</p>"
    ) is False
    assert _html_is_gone(
        "<title>Engineer</title><p>This job expired on January 1.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title><p>This job posting expired on March 1, 2026.</p>"
        "<p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title><p>Sorry, this job expired.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title><p>This listing expired.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title><p>This job expires on January 1.</p><p>$180,000</p>"
    ) is False
    assert _html_is_gone(
        "<title>Engineer</title><p>This posting will expire on Friday.</p>"
        "<p>$180,000</p>"
    ) is False
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>Applications for this position are closed.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>Applications for this role are now closed.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title><p>We are no longer taking applications.</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>This role is no longer open to applicants.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>This position is no longer open for applications.</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title><p>This role is no longer open.</p><p>$180,000</p>"
    ) is False
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>This visa is no longer open to contractors.</p><p>$180,000</p>"
    ) is False
    assert _html_is_gone(
        "<title>Engineer</title><p>This posting is expired.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title><p>This req is closed.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title><p>The req has been filled.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title><p>This req has expired.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title><p>This req is no longer available.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title><p>This request is closed.</p><p>$180,000</p>"
    ) is False
    assert _html_is_gone(
        "<title>Engineer</title><p>Position has expired.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title><p>Job has been filled.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title><p>Position has been filled.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title><p>Sorry, position has expired.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title><p>Position expired.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>Once this position has been filled, the team will grow.</p>"
        "<p>$180,000 a year</p>"
    ) is False
    assert _html_is_gone(
        "<title>Engineer</title><p>This job expires on January 1.</p><p>$180,000</p>"
    ) is False
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>The application window has closed.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title><p>Application period is closed.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>The application window closes on Friday.</p><p>$180,000</p>"
    ) is False
    assert _html_is_gone(
        "<title>Engineer</title><p>Applications close on January 1.</p><p>$180,000</p>"
    ) is False
    assert _html_is_gone(
        "<title>Engineer</title><p>Applications are now closed.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title><p>This application is closed.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>This job is no longer accepting applicants.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>The application deadline has passed.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title><p>This request is closed.</p><p>$180,000</p>"
    ) is False
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>Once applications are closed, the team will follow up.</p>"
        "<p>$180,000</p>"
    ) is False
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>When this application is closed, recruiters will reach out.</p>"
        "<p>$180,000</p>"
    ) is False
    assert _html_is_gone(
        "<title>Engineer</title><p>Position is no longer available.</p>"
        "<p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title><p>Job is no longer available.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title><p>Role is no longer active.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title><p>Listing is no longer posted.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>Sorry, position is no longer available.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>Once this position is no longer available, we will archive it.</p>"
        "<p>$180,000</p>"
    ) is False
    assert _html_is_gone(
        "<title>Engineer</title><p>This role has ended.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title><p>This position has ended.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title><p>The role has ended.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title><p>This job has ended.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title><p>Position has ended.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title><p>This role ended.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>Once this role has ended, we will archive it.</p>"
        "<p>$180,000</p>"
    ) is False
    assert _html_is_gone(
        "<title>Engineer</title><p>This role will end next quarter.</p>"
        "<p>$180,000</p>"
    ) is False
    assert _html_is_gone(
        "<title>Engineer</title><p>This role ends on Friday.</p><p>$180,000</p>"
    ) is False
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>The role ended up being remote.</p><p>$180,000</p>"
    ) is False
    assert _html_is_gone(
        "<title>Engineer</title><p>Job closed.</p><p>$180,000</p>"
    ) is False
    assert _html_is_gone(
        "<title>Engineer</title><p>Position closed.</p><p>$180,000</p>"
    ) is False
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>This posting has been discontinued.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>This job has been discontinued.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title><p>Position has been removed.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>This position has been removed.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>Once this position has been removed, we will archive it.</p>"
        "<p>$180,000</p>"
    ) is False
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>Applications removed from consideration stay on file.</p>"
        "<p>$180,000</p>"
    ) is False
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>This posting has been archived.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>This job has been archived.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>This requisition has been archived.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>Once this posting has been archived, we will notify you.</p>"
        "<p>$180,000</p>"
    ) is False
    assert _html_is_gone(
        "<title>Engineer</title><p>This job doesn't exist.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title><p>This posting no longer exists.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title><p>This vacancy no longer exists.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title><p>Position no longer exists.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title><p>This role is no longer live.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>This role will not exist until next year.</p><p>$180,000</p>"
    ) is False
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>Sorry, we couldn't find this job.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>This job could not be found.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>The job you are looking for is no longer available.</p>"
        "<p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>We couldn't find this job description in the PDF.</p>"
        "<p>$180,000</p>"
    ) is False
    assert _html_is_gone(
        "<title>Engineer</title><p>Job not found.</p><p>$180,000</p>"
    ) is False
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>This job has been unpublished.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>This posting has been unpublished.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>Applications for this role have closed.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>This role is no longer accepting new applicants.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>Once this job has been unpublished, we will archive it.</p>"
        "<p>$180,000</p>"
    ) is False
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>This job is no longer published.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>This posting is unpublished.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>This search has been closed.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>We've decided not to fill this role.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>This job posting was taken down.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>Once this search has been closed, we will archive it.</p>"
        "<p>$180,000</p>"
    ) is False
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>We are no longer recruiting for this role.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>Applications are no longer being accepted.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>This job has been taken offline.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>We will not be filling this role.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>This role is no longer open.</p><p>$180,000</p>"
    ) is False
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>We've paused hiring for this role.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>This job has been put on hold.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>We are not currently hiring for this role.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>Position is no longer open for applications.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>This posting is inactive.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>This job has been deactivated.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>We've withdrawn this role.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>Sorry, we could not find this job.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>We have cancelled this vacancy.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>The posting you requested no longer exists.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>No longer accepting candidates.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>We are no longer reviewing applications.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>This role is no longer open.</p><p>$180,000</p>"
    ) is False
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>This job has been unposted.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>This posting has been deleted.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>We've removed this posting.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>This posting is archived.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>We're not accepting applications for this role.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>Position filled.</p><p>$180,000</p>"
    ) is False
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>This page has been removed.</p><p>$180,000</p>"
    ) is False
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>We've stopped accepting applications.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>Hiring for this role has ended.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>This role is no longer being recruited.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>This search has concluded.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>No longer hiring for this position.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>We've stopped accepting applications from recruiters.</p><p>$180,000</p>"
    ) is False
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>We've unpublished this role.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>This position was unpublished.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>We've taken this job down.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>We are no longer filling this role.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>This position is no longer being filled.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>The application window has ended.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>Applications closed for this role.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>This role is no longer open.</p><p>$180,000</p>"
    ) is False
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>This job is gone.</p><p>$180,000</p>"
    ) is False
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>We are no longer recruiting this role.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>We've closed recruiting for this role.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>We could not locate this job.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>This search expired.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>The job you selected could not be found.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>couldn't find this job description.</p><p>$180,000</p>"
    ) is False
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>Applications are no longer open.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>This job posting cannot be found.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>This visa is no longer open to contractors.</p><p>$180,000</p>"
    ) is False
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>We filled this role.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>Hiring has closed for this role.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>We were unable to find this job.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>The position you applied for is no longer available.</p>"
        "<p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>The role you applied for no longer exists.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>Applications are no longer being reviewed.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>This posting was deleted.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>This job is deactivated.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>We've paused this search.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>This search is on hold.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>Recruiting has ended for this role.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>The application window ended.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>We've stopped taking applications.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>We've stopped taking applications from recruiters.</p>"
        "<p>$180,000</p>"
    ) is False
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>Applications closed for this search.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>We could not find this opportunity.</p><p>$180,000</p>"
    ) is True
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>We were unable to find this job description.</p><p>$180,000</p>"
    ) is False
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>This role is no longer open.</p><p>$180,000</p>"
    ) is False
    assert _html_is_gone(
        "<title>Engineer</title>"
        "<p>This job is gone.</p><p>$180,000</p>"
    ) is False


def test_listing_text_removed_html_is_gone(monkeypatch):
    engine = Engine()

    async def fake_get(_client, url: str):
        return (
            "<title>ML Engineer - Zoom | Built In</title>"
            "<p>Sorry, this job was removed at 08:39 a.m. (UTC)</p>"
            "<p>$151,800 - $332,200</p>"
        )

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    html = asyncio.run(
        engine._listing_text(
            "https://www.builtin.com/job/machine-learning-engineer-agentic-retrieval/10377512"
        )
    )
    assert html is None


def test_enrich_drops_removed_listing_html():
    engine = Engine()

    async def page(url: str):
        if "removed" in url:
            return (
                "<title>ML Engineer - Zoom | Built In</title>"
                "<p>Sorry, this job was removed at 08:39 a.m. (UTC)</p>"
                "<p>$151,800</p>"
            )
        return "<title>Senior ML</title><p>$180,000 a year</p>"

    engine._listing_text = page
    keep = Opportunity(title="Keep", url="https://jobs.example/live")
    ghost = Opportunity(title="Ghost", url="https://www.builtin.com/job/removed/1")
    opps = [keep, ghost]
    asyncio.run(engine._enrich_pay(opps))
    assert [o.title for o in opps] == ["Keep"]
    assert keep.pay_high == 180_000


def test_enrich_drops_filled_listing_with_leftover_pay():
    engine = Engine()

    async def page(url: str):
        if "filled" in url:
            return (
                "<title>Senior ML Engineer</title>"
                "<p>This position has been filled.</p>"
                "<p>$180,000 - $220,000 a year</p>"
            )
        return "<title>Senior ML</title><p>$180,000 a year</p>"

    engine._listing_text = page
    keep = Opportunity(title="Keep", url="https://jobs.example/live")
    ghost = Opportunity(
        title="Ghost",
        url="https://jobs.example/filled",
        pay_high=220_000,
        hours_per_week=40,
    )
    opps = [keep, ghost]
    asyncio.run(engine._enrich_pay(opps))
    assert [o.title for o in opps] == ["Keep"]
    assert keep.pay_high == 180_000


def test_enrich_drops_closed_opportunity_with_leftover_pay():
    engine = Engine()

    async def page(url: str):
        if "closed" in url:
            return (
                "<title>Senior ML Engineer</title>"
                "<p>This opportunity is no longer available.</p>"
                "<p>$180,000 - $220,000 a year</p>"
            )
        return "<title>Senior ML</title><p>$180,000 a year</p>"

    engine._listing_text = page
    keep = Opportunity(title="Keep", url="https://jobs.example/live")
    ghost = Opportunity(
        title="Ghost",
        url="https://jobs.example/closed",
        pay_high=220_000,
        hours_per_week=40,
    )
    opps = [keep, ghost]
    asyncio.run(engine._enrich_pay(opps))
    assert [o.title for o in opps] == ["Keep"]
    assert keep.pay_high == 180_000


def test_enrich_drops_withdrawn_listing_with_leftover_pay():
    engine = Engine()

    async def page(url: str):
        if "withdrawn" in url:
            return (
                "<title>Senior ML Engineer</title>"
                "<p>This listing is no longer available.</p>"
                "<p>$180,000 - $220,000 a year</p>"
            )
        return "<title>Senior ML</title><p>$180,000 a year</p>"

    engine._listing_text = page
    keep = Opportunity(title="Keep", url="https://jobs.example/live")
    ghost = Opportunity(
        title="Ghost",
        url="https://jobs.example/withdrawn",
        pay_high=220_000,
        hours_per_week=40,
    )
    opps = [keep, ghost]
    asyncio.run(engine._enrich_pay(opps))
    assert [o.title for o in opps] == ["Keep"]
    assert keep.pay_high == 180_000


def test_enrich_drops_expired_listing_with_leftover_pay():
    engine = Engine()

    async def page(url: str):
        if "no-longer-avail" in url:
            return (
                "<title>Senior ML Engineer</title>"
                "<p>Position is no longer available.</p>"
                "<p>$180,000 - $220,000 a year</p>"
            )
        if "role-ended" in url:
            return (
                "<title>Senior ML Engineer</title>"
                "<p>This role has ended.</p>"
                "<p>$180,000 - $220,000 a year</p>"
            )
        if "discontinued" in url:
            return (
                "<title>Senior ML Engineer</title>"
                "<p>This posting has been discontinued.</p>"
                "<p>$180,000 - $220,000 a year</p>"
            )
        if "been-removed" in url:
            return (
                "<title>Senior ML Engineer</title>"
                "<p>Position has been removed.</p>"
                "<p>$180,000 - $220,000 a year</p>"
            )
        if "archived" in url:
            return (
                "<title>Senior ML Engineer</title>"
                "<p>This posting has been archived.</p>"
                "<p>$180,000 - $220,000 a year</p>"
            )
        if "doesnt-exist" in url:
            return (
                "<title>Senior ML Engineer</title>"
                "<p>This job doesn't exist.</p>"
                "<p>$180,000 - $220,000 a year</p>"
            )
        if "couldnt-find" in url:
            return (
                "<title>Senior ML Engineer</title>"
                "<p>Sorry, we couldn't find this job.</p>"
                "<p>$180,000 - $220,000 a year</p>"
            )
        if "looking-for" in url:
            return (
                "<title>Senior ML Engineer</title>"
                "<p>The job you are looking for is no longer available.</p>"
                "<p>$180,000 - $220,000 a year</p>"
            )
        if "is-unpublished" in url:
            return (
                "<title>Senior ML Engineer</title>"
                "<p>This posting is unpublished.</p>"
                "<p>$180,000 - $220,000 a year</p>"
            )
        if "unpublished" in url:
            return (
                "<title>Senior ML Engineer</title>"
                "<p>This posting has been unpublished.</p>"
                "<p>$180,000 - $220,000 a year</p>"
            )
        if "have-closed" in url:
            return (
                "<title>Senior ML Engineer</title>"
                "<p>Applications for this role have closed.</p>"
                "<p>$180,000 - $220,000 a year</p>"
            )
        if "new-apps" in url:
            return (
                "<title>Senior ML Engineer</title>"
                "<p>This role is no longer accepting new applicants.</p>"
                "<p>$180,000 - $220,000 a year</p>"
            )
        if "no-longer-pub" in url:
            return (
                "<title>Senior ML Engineer</title>"
                "<p>This job is no longer published.</p>"
                "<p>$180,000 - $220,000 a year</p>"
            )
        if "search-closed" in url:
            return (
                "<title>Senior ML Engineer</title>"
                "<p>This search has been closed.</p>"
                "<p>$180,000 - $220,000 a year</p>"
            )
        if "not-fill" in url:
            return (
                "<title>Senior ML Engineer</title>"
                "<p>We've decided not to fill this role.</p>"
                "<p>$180,000 - $220,000 a year</p>"
            )
        if "taken-down" in url:
            return (
                "<title>Senior ML Engineer</title>"
                "<p>This job posting was taken down.</p>"
                "<p>$180,000 - $220,000 a year</p>"
            )
        if "recruiting" in url:
            return (
                "<title>Senior ML Engineer</title>"
                "<p>We are no longer recruiting for this role.</p>"
                "<p>$180,000 - $220,000 a year</p>"
            )
        if "being-accepted" in url:
            return (
                "<title>Senior ML Engineer</title>"
                "<p>Applications are no longer being accepted.</p>"
                "<p>$180,000 - $220,000 a year</p>"
            )
        if "offline" in url:
            return (
                "<title>Senior ML Engineer</title>"
                "<p>This job has been taken offline.</p>"
                "<p>$180,000 - $220,000 a year</p>"
            )
        if "paused" in url:
            return (
                "<title>Senior ML Engineer</title>"
                "<p>We've paused hiring for this role.</p>"
                "<p>$180,000 - $220,000 a year</p>"
            )
        if "on-hold" in url:
            return (
                "<title>Senior ML Engineer</title>"
                "<p>This job has been put on hold.</p>"
                "<p>$180,000 - $220,000 a year</p>"
            )
        if "inactive" in url:
            return (
                "<title>Senior ML Engineer</title>"
                "<p>This posting is inactive.</p>"
                "<p>$180,000 - $220,000 a year</p>"
            )
        if "is-expired" in url:
            return (
                "<title>Senior ML Engineer</title>"
                "<p>This posting is expired.</p>"
                "<p>$180,000 - $220,000 a year</p>"
            )
        if "position-expired" in url:
            return (
                "<title>Senior ML Engineer</title>"
                "<p>Position has expired.</p>"
                "<p>$180,000 - $220,000 a year</p>"
            )
        if "job-filled" in url:
            return (
                "<title>Senior ML Engineer</title>"
                "<p>Job has been filled.</p>"
                "<p>$180,000 - $220,000 a year</p>"
            )
        if "req-closed" in url:
            return (
                "<title>Senior ML Engineer</title>"
                "<p>This req is closed.</p>"
                "<p>$180,000 - $220,000 a year</p>"
            )
        if "expired" in url:
            return (
                "<title>Senior ML Engineer</title>"
                "<p>This job expired on January 1.</p>"
                "<p>$180,000 - $220,000 a year</p>"
            )
        if "apps-closed" in url:
            return (
                "<title>Senior ML Engineer</title>"
                "<p>Applications for this position are closed.</p>"
                "<p>$180,000 - $220,000 a year</p>"
            )
        if "window-closed" in url:
            return (
                "<title>Senior ML Engineer</title>"
                "<p>The application window has closed.</p>"
                "<p>$180,000 - $220,000 a year</p>"
            )
        if "now-closed" in url:
            return (
                "<title>Senior ML Engineer</title>"
                "<p>Applications are now closed.</p>"
                "<p>$180,000 - $220,000 a year</p>"
            )
        if "application-closed" in url:
            return (
                "<title>Senior ML Engineer</title>"
                "<p>This application is closed.</p>"
                "<p>$180,000 - $220,000 a year</p>"
            )
        if "applicants" in url:
            return (
                "<title>Senior ML Engineer</title>"
                "<p>This job is no longer accepting applicants.</p>"
                "<p>$180,000 - $220,000 a year</p>"
            )
        if "deadline-passed" in url:
            return (
                "<title>Senior ML Engineer</title>"
                "<p>The application deadline has passed.</p>"
                "<p>$180,000 - $220,000 a year</p>"
            )
        return "<title>Senior ML</title><p>$180,000 a year</p>"

    engine._listing_text = page
    keep = Opportunity(title="Keep", url="https://jobs.example/live")
    expired = Opportunity(
        title="Expired",
        url="https://jobs.example/expired",
        pay_high=220_000,
        hours_per_week=40,
    )
    closed = Opportunity(
        title="ClosedApps",
        url="https://jobs.example/apps-closed",
        pay_high=220_000,
        hours_per_week=40,
    )
    window = Opportunity(
        title="WindowClosed",
        url="https://jobs.example/window-closed",
        pay_high=220_000,
        hours_per_week=40,
    )
    is_expired = Opportunity(
        title="IsExpired",
        url="https://jobs.example/is-expired",
        pay_high=220_000,
        hours_per_week=40,
    )
    position_expired = Opportunity(
        title="PositionExpired",
        url="https://jobs.example/position-expired",
        pay_high=220_000,
        hours_per_week=40,
    )
    job_filled = Opportunity(
        title="JobFilled",
        url="https://jobs.example/job-filled",
        pay_high=220_000,
        hours_per_week=40,
    )
    req_closed = Opportunity(
        title="ReqClosed",
        url="https://jobs.example/req-closed",
        pay_high=220_000,
        hours_per_week=40,
    )
    now_closed = Opportunity(
        title="NowClosed",
        url="https://jobs.example/now-closed",
        pay_high=220_000,
        hours_per_week=40,
    )
    application_closed = Opportunity(
        title="ApplicationClosed",
        url="https://jobs.example/application-closed",
        pay_high=220_000,
        hours_per_week=40,
    )
    applicants = Opportunity(
        title="Applicants",
        url="https://jobs.example/applicants",
        pay_high=220_000,
        hours_per_week=40,
    )
    deadline = Opportunity(
        title="DeadlinePassed",
        url="https://jobs.example/deadline-passed",
        pay_high=220_000,
        hours_per_week=40,
    )
    no_longer_avail = Opportunity(
        title="NoLongerAvail",
        url="https://jobs.example/no-longer-avail",
        pay_high=220_000,
        hours_per_week=40,
    )
    role_ended = Opportunity(
        title="RoleEnded",
        url="https://jobs.example/role-ended",
        pay_high=220_000,
        hours_per_week=40,
    )
    discontinued = Opportunity(
        title="Discontinued",
        url="https://jobs.example/discontinued",
        pay_high=220_000,
        hours_per_week=40,
    )
    been_removed = Opportunity(
        title="BeenRemoved",
        url="https://jobs.example/been-removed",
        pay_high=220_000,
        hours_per_week=40,
    )
    archived = Opportunity(
        title="Archived",
        url="https://jobs.example/archived",
        pay_high=220_000,
        hours_per_week=40,
    )
    doesnt_exist = Opportunity(
        title="DoesntExist",
        url="https://jobs.example/doesnt-exist",
        pay_high=220_000,
        hours_per_week=40,
    )
    couldnt_find = Opportunity(
        title="CouldntFind",
        url="https://jobs.example/couldnt-find",
        pay_high=220_000,
        hours_per_week=40,
    )
    looking_for = Opportunity(
        title="LookingFor",
        url="https://jobs.example/looking-for",
        pay_high=220_000,
        hours_per_week=40,
    )
    unpublished = Opportunity(
        title="Unpublished",
        url="https://jobs.example/unpublished",
        pay_high=220_000,
        hours_per_week=40,
    )
    have_closed = Opportunity(
        title="HaveClosed",
        url="https://jobs.example/have-closed",
        pay_high=220_000,
        hours_per_week=40,
    )
    new_apps = Opportunity(
        title="NewApps",
        url="https://jobs.example/new-apps",
        pay_high=220_000,
        hours_per_week=40,
    )
    no_longer_pub = Opportunity(
        title="NoLongerPub",
        url="https://jobs.example/no-longer-pub",
        pay_high=220_000,
        hours_per_week=40,
    )
    is_unpublished = Opportunity(
        title="IsUnpublished",
        url="https://jobs.example/is-unpublished",
        pay_high=220_000,
        hours_per_week=40,
    )
    search_closed = Opportunity(
        title="SearchClosed",
        url="https://jobs.example/search-closed",
        pay_high=220_000,
        hours_per_week=40,
    )
    not_fill = Opportunity(
        title="NotFill",
        url="https://jobs.example/not-fill",
        pay_high=220_000,
        hours_per_week=40,
    )
    taken_down = Opportunity(
        title="TakenDown",
        url="https://jobs.example/taken-down",
        pay_high=220_000,
        hours_per_week=40,
    )
    recruiting = Opportunity(
        title="Recruiting",
        url="https://jobs.example/recruiting",
        pay_high=220_000,
        hours_per_week=40,
    )
    being_accepted = Opportunity(
        title="BeingAccepted",
        url="https://jobs.example/being-accepted",
        pay_high=220_000,
        hours_per_week=40,
    )
    offline = Opportunity(
        title="Offline",
        url="https://jobs.example/offline",
        pay_high=220_000,
        hours_per_week=40,
    )
    paused = Opportunity(
        title="Paused",
        url="https://jobs.example/paused",
        pay_high=220_000,
        hours_per_week=40,
    )
    on_hold = Opportunity(
        title="OnHold",
        url="https://jobs.example/on-hold",
        pay_high=220_000,
        hours_per_week=40,
    )
    inactive = Opportunity(
        title="Inactive",
        url="https://jobs.example/inactive",
        pay_high=220_000,
        hours_per_week=40,
    )
    opps = [
        keep,
        expired,
        closed,
        window,
        is_expired,
        position_expired,
        job_filled,
        req_closed,
        now_closed,
        application_closed,
        applicants,
        deadline,
        no_longer_avail,
        role_ended,
        discontinued,
        been_removed,
        archived,
        doesnt_exist,
        couldnt_find,
        looking_for,
        unpublished,
        have_closed,
        new_apps,
        no_longer_pub,
        is_unpublished,
        search_closed,
        not_fill,
        taken_down,
        recruiting,
        being_accepted,
        offline,
        paused,
        on_hold,
        inactive,
    ]
    asyncio.run(engine._enrich_pay(opps))
    assert [o.title for o in opps] == ["Keep"]
    assert keep.pay_high == 180_000


def test_enrich_drops_fetched_board_index_html():
    engine = Engine()

    async def page(url: str) -> str:
        if "grafanalabs" in url:
            return "<title>Jobs at Grafana Labs</title><p>Current openings</p>"
        if "open-positions" in url:
            return "<title>Open Positions | Stripe</title><p>$180,000 - $270,000</p>"
        if "life-at" in url:
            return "<title>Life at Acme</title><p>$180,000 - $270,000</p>"
        if "internships" in url:
            return "<title>Internships | Acme</title><p>$180,000</p>"
        if "meet-the-team" in url:
            return "<title>Meet the Team | Acme</title><p>$180,000 - $270,000</p>"
        if "campus-recruiting" in url:
            return "<title>Campus Recruiting | Acme</title><p>$180,000 - $270,000</p>"
        if "early-careers" in url:
            return "<title>Early Careers | Acme</title><p>$180,000</p>"
        if "job-search" in url:
            return "<title>Job Search | Acme</title><p>$180,000 - $270,000</p>"
        if "careers-overview" in url:
            return "<title>Careers | Acme</title><p>$180,000 - $270,000</p>"
        if "our-benefits" in url:
            return "<title>Our Benefits</title><p>$180,000</p>"
        if "our-culture" in url:
            return "<title>Our Culture</title><p>$180,000</p>"
        if url.rstrip("/").endswith("/culture"):
            return "<title>Culture | Acme</title><p>$180,000 - $270,000</p>"
        if url.rstrip("/").endswith("/leadership"):
            return "<title>Leadership | Acme</title><p>$180,000</p>"
        if "about-us" in url:
            return "<title>About Us | Acme</title><p>$180,000 - $270,000</p>"
        if "our-values" in url:
            return "<title>Our Values | Acme</title><p>$180,000</p>"
        if url.rstrip("/").endswith("/locations"):
            return "<title>Locations | Acme</title><p>$180,000 - $270,000</p>"
        if url.rstrip("/").endswith("/diversity"):
            return "<title>Diversity | Acme</title><p>$180,000 - $270,000</p>"
        if url.rstrip("/").endswith("/dei"):
            return "<title>DEI | Acme</title><p>$180,000</p>"
        if "our-story" in url:
            return "<title>Our Story | Acme</title><p>$180,000 - $270,000</p>"
        if url.rstrip("/").endswith("/faqs"):
            return "<title>FAQs | Acme</title><p>$180,000</p>"
        if url.rstrip("/").endswith("/news"):
            return "<title>News | Acme</title><p>$180,000 - $270,000</p>"
        if url.rstrip("/").endswith("/newsroom"):
            return "<title>Newsroom | Acme</title><p>$180,000 - $270,000</p>"
        if url.rstrip("/").endswith("/investors"):
            return "<title>Investors | Acme</title><p>$180,000</p>"
        if url.rstrip("/").endswith("/sustainability"):
            return "<title>Sustainability | Acme</title><p>$180,000</p>"
        if url.rstrip("/").endswith("/esg"):
            return "<title>ESG | Acme</title><p>$180,000</p>"
        if url.rstrip("/").endswith("/impact"):
            return "<title>Impact | Acme</title><p>$180,000 - $270,000</p>"
        if url.rstrip("/").endswith("/community"):
            return "<title>Community | Acme</title><p>$180,000</p>"
        if url.rstrip("/").endswith("/csr"):
            return "<title>CSR | Acme</title><p>$180,000</p>"
        if url.rstrip("/").endswith("/purpose"):
            return "<title>Purpose | Acme</title><p>$180,000</p>"
        if url.rstrip("/").endswith("/people"):
            return "<title>People | Acme</title><p>$180,000</p>"
        if url.rstrip("/").endswith("/ethics"):
            return "<title>Ethics | Acme</title><p>$180,000</p>"
        if url.rstrip("/").endswith("/media-center"):
            return "<title>Media Center | Acme</title><p>$180,000</p>"
        if url.rstrip("/").endswith("/environment"):
            return "<title>Environment | Acme</title><p>$180,000</p>"
        if url.rstrip("/").endswith("/foundation"):
            return "<title>Foundation | Acme</title><p>$180,000</p>"
        if url.rstrip("/").endswith("/giving"):
            return "<title>Giving | Acme</title><p>$180,000</p>"
        if url.rstrip("/").endswith("/philanthropy"):
            return "<title>Philanthropy | Acme</title><p>$180,000</p>"
        if url.rstrip("/").endswith("/citizenship"):
            return "<title>Citizenship | Acme</title><p>$180,000</p>"
        if url.rstrip("/").endswith("/charity"):
            return "<title>Charity | Acme</title><p>$180,000</p>"
        if url.rstrip("/").endswith("/responsibility"):
            return "<title>Responsibility | Acme</title><p>$180,000</p>"
        return ""

    engine._listing_text = page
    keep = Opportunity(title="Real", url="https://jobs.example/real", pay_high=100_000, company="Acme")
    ghost = Opportunity(
        title="Senior Machine Learning Engineer, Developer Advocacy | US | Remote",
        url="https://job-boards.greenhouse.io/grafanalabs/jobs/1",
    )
    catalog = Opportunity(
        title="Open Positions | Stripe",
        url="https://stripe.com/jobs/open-positions",
        pay_high=270_000,
    )
    life = Opportunity(
        title="Life at Acme",
        url="https://acme.com/life-at-acme",
        pay_high=270_000,
    )
    internships = Opportunity(
        title="Internships | Acme",
        url="https://acme.com/internships",
        pay_high=180_000,
    )
    meet = Opportunity(
        title="Meet the Team | Acme",
        url="https://acme.com/meet-the-team",
        pay_high=270_000,
    )
    campus = Opportunity(
        title="Campus Recruiting | Acme",
        url="https://acme.com/campus-recruiting",
        pay_high=270_000,
    )
    early = Opportunity(
        title="Early Careers | Acme",
        url="https://acme.com/early-careers",
        pay_high=180_000,
    )
    job_search = Opportunity(
        title="Job Search | Acme",
        url="https://acme.com/job-search",
        pay_high=270_000,
    )
    careers = Opportunity(
        title="Careers | Acme",
        url="https://acme.com/about/careers-overview",
        pay_high=270_000,
    )
    benefits = Opportunity(
        title="Our Benefits",
        url="https://acme.com/our-benefits",
        pay_high=180_000,
    )
    culture = Opportunity(
        title="Culture | Acme",
        url="https://acme.com/culture",
        pay_high=270_000,
    )
    leadership = Opportunity(
        title="Leadership | Acme",
        url="https://acme.com/leadership",
        pay_high=180_000,
    )
    about = Opportunity(
        title="About Us | Acme",
        url="https://acme.com/about-us",
        pay_high=270_000,
    )
    values = Opportunity(
        title="Our Values | Acme",
        url="https://acme.com/our-values",
        pay_high=180_000,
    )
    locations = Opportunity(
        title="Locations | Acme",
        url="https://acme.com/locations",
        pay_high=270_000,
    )
    diversity = Opportunity(
        title="Diversity | Acme",
        url="https://acme.com/diversity",
        pay_high=270_000,
    )
    dei = Opportunity(
        title="DEI | Acme",
        url="https://acme.com/dei",
        pay_high=180_000,
    )
    story = Opportunity(
        title="Our Story | Acme",
        url="https://acme.com/our-story",
        pay_high=270_000,
    )
    faqs = Opportunity(
        title="FAQs | Acme",
        url="https://acme.com/faqs",
        pay_high=180_000,
    )
    news = Opportunity(
        title="News | Acme",
        url="https://acme.com/news",
        pay_high=270_000,
    )
    newsroom = Opportunity(
        title="Newsroom | Acme",
        url="https://acme.com/newsroom",
        pay_high=270_000,
    )
    investors = Opportunity(
        title="Investors | Acme",
        url="https://acme.com/investors",
        pay_high=180_000,
    )
    sustainability = Opportunity(
        title="Sustainability | Acme",
        url="https://acme.com/sustainability",
        pay_high=180_000,
    )
    esg = Opportunity(
        title="ESG | Acme",
        url="https://acme.com/esg",
        pay_high=180_000,
    )
    impact = Opportunity(
        title="Impact | Acme",
        url="https://acme.com/impact",
        pay_high=270_000,
    )
    community = Opportunity(
        title="Community | Acme",
        url="https://acme.com/community",
        pay_high=180_000,
    )
    csr = Opportunity(
        title="CSR | Acme",
        url="https://acme.com/csr",
        pay_high=180_000,
    )
    purpose = Opportunity(
        title="Purpose | Acme",
        url="https://acme.com/purpose",
        pay_high=180_000,
    )
    people = Opportunity(
        title="People | Acme",
        url="https://acme.com/people",
        pay_high=180_000,
    )
    ethics = Opportunity(
        title="Ethics | Acme",
        url="https://acme.com/ethics",
        pay_high=180_000,
    )
    media_center = Opportunity(
        title="Media Center | Acme",
        url="https://acme.com/media-center",
        pay_high=180_000,
    )
    environment = Opportunity(
        title="Environment | Acme",
        url="https://acme.com/environment",
        pay_high=180_000,
    )
    foundation = Opportunity(
        title="Foundation | Acme",
        url="https://acme.com/foundation",
        pay_high=180_000,
    )
    giving = Opportunity(
        title="Giving | Acme",
        url="https://acme.com/giving",
        pay_high=180_000,
    )
    philanthropy = Opportunity(
        title="Philanthropy | Acme",
        url="https://acme.com/philanthropy",
        pay_high=180_000,
    )
    citizenship = Opportunity(
        title="Citizenship | Acme",
        url="https://acme.com/citizenship",
        pay_high=180_000,
    )
    charity = Opportunity(
        title="Charity | Acme",
        url="https://acme.com/charity",
        pay_high=180_000,
    )
    responsibility = Opportunity(
        title="Responsibility | Acme",
        url="https://acme.com/responsibility",
        pay_high=180_000,
    )
    opps = [keep, ghost, catalog, life, internships, meet, campus, early, job_search, careers, benefits, culture, leadership, about, values, locations, diversity, dei, story, faqs, news, newsroom, investors, sustainability, esg, impact, community, csr, purpose, people, ethics, media_center, environment, foundation, giving, philanthropy, citizenship, charity, responsibility]
    asyncio.run(engine._enrich_pay(opps))
    assert [o.title for o in opps] == ["Real"]


def test_http_get_text_none_on_404_empty_on_403():
    from src.engine import _http_get_text

    class _Resp:
        def __init__(self, status: int, body: str):
            self.status_code = status
            self.text = body

    class _Client:
        def __init__(self, status: int):
            self.status = status

        async def get(self, _url: str):
            return _Resp(self.status, "x" * 1000)

    assert asyncio.run(_http_get_text(_Client(404), "https://jobs.lever.co/x")) is None
    assert asyncio.run(_http_get_text(_Client(410), "https://jobs.lever.co/x")) is None
    assert asyncio.run(_http_get_text(_Client(403), "https://jobs.lever.co/x")) == ""


def test_http_get_text_cloudflare_challenge_is_gone():
    from src.engine import _cloudflare_challenge, _html_is_index, _http_get_text

    cf = (
        "<!DOCTYPE html><html lang=\"en-US\"><head>"
        "<title>Just a moment...</title></head><body>"
        "<script src=\"/cdn-cgi/challenge-platform/h/b/orchestrate/jsch/v1\"></script>"
        "</body></html>"
    )
    assert _cloudflare_challenge(cf) is True
    assert _html_is_index(cf, "https://jobs.uber.com/en/jobs/145860/") is True

    class _Resp:
        def __init__(self, status: int, body: str):
            self.status_code = status
            self.text = body

    class _Client:
        def __init__(self, status: int, body: str):
            self.status = status
            self.body = body

        async def get(self, _url: str):
            return _Resp(self.status, self.body)

    assert (
        asyncio.run(_http_get_text(_Client(403, cf), "https://jobs.uber.com/en/jobs/145860/"))
        is None
    )
    assert (
        asyncio.run(_http_get_text(_Client(200, cf), "https://jobs.uber.com/en/jobs/145860/"))
        is None
    )
    denied = '{"errorCode":"S22","httpStatus":403,"message":"permission denied"}'
    assert _cloudflare_challenge(denied) is False
    assert (
        asyncio.run(
            _http_get_text(_Client(403, denied), "https://shipt.wd1.myworkdayjobs.com/wday/cxs/x")
        )
        == ""
    )


def test_listing_text_none_when_canonical_page_is_gone(monkeypatch):
    engine = Engine()

    async def fake_get(_client, _url: str):
        return None

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    html = asyncio.run(
        engine._listing_text(
            "https://jobs.lever.co/provectus/76640225-4aa7-45a3-bcdc-cb156271057b"
        )
    )
    assert html is None


def test_listing_text_lever_api_404_is_gone(monkeypatch):
    engine = Engine()
    seen: list[str] = []

    async def fake_get(_client, url: str):
        seen.append(url)
        if "api.lever.co" in url:
            return None
        return "<title>Jobs at Provectus</title><p>Current openings</p>"

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    html = asyncio.run(
        engine._listing_text(
            "https://jobs.lever.co/provectus/76640225-4aa7-45a3-bcdc-cb156271057b"
        )
    )
    assert seen == [
        "https://api.lever.co/v0/postings/provectus/76640225-4aa7-45a3-bcdc-cb156271057b"
    ]
    assert html is None


def test_ashby_ids_strips_application():
    from src.engine import _ashby_ids

    assert _ashby_ids(
        "https://jobs.ashbyhq.com/quilter/9a15ed0b-1a0e-4c00-b7c8-8a0c4e8e9abc/application"
    ) == ("quilter", "9a15ed0b-1a0e-4c00-b7c8-8a0c4e8e9abc")
    assert _ashby_ids("https://jobs.ashbyhq.com/quilter") is None


def test_ashby_to_html_pay_from_scrapeable_summary():
    from src.engine import _apply_listing, _ashby_to_html

    html = _ashby_to_html(
        {
            "title": "Machine Learning Engineer",
            "employmentType": "FullTime",
            "workplaceType": "Remote",
            "descriptionHtml": "<p>Build ML systems.</p>",
            "scrapeableCompensationSalarySummary": "$180K - $200K",
        }
    )
    opp = Opportunity(
        title="x",
        url="https://jobs.ashbyhq.com/quilter/9a15ed0b-1a0e-4c00-b7c8-8a0c4e8e9abc",
    )
    _apply_listing(opp, html)
    assert opp.title == "Machine Learning Engineer"
    assert opp.company == "Quilter"
    assert opp.pay_low == 180_000
    assert opp.pay_high == 200_000
    assert opp.hours_per_week == 40
    assert opp.rate_is_imputed is False
    assert opp.remote is True
    assert opp.score() == 100


def test_ashby_to_html_city_location_is_office_when_workplace_missing():
    from src.engine import _ASHBY_JOB_QUERY, _apply_listing, _ashby_to_html

    assert "locationName" in _ASHBY_JOB_QUERY
    html = _ashby_to_html(
        {
            "title": "Staff Machine Learning Engineer",
            "employmentType": "FullTime",
            "workplaceType": None,
            "locationName": "Austin, TX",
            "descriptionHtml": "<p>Build ML systems. Competitive salary.</p>",
        }
    )
    opp = Opportunity(
        title="x",
        url="https://jobs.ashbyhq.com/webAI/3f4040c2-e8c4-4e52-b3d5-be0d02e5c6b3",
        remote=True,
    )
    _apply_listing(opp, html)
    assert opp.remote is False
    assert "Austin, TX" in html


def test_ashby_to_html_remote_location_name_when_workplace_missing():
    from src.engine import _apply_listing, _ashby_to_html

    html = _ashby_to_html(
        {
            "title": "Engineer",
            "employmentType": "FullTime",
            "locationName": "Remote, United States",
            "descriptionHtml": "<p>Build systems.</p>",
        }
    )
    opp = Opportunity(
        title="x",
        url="https://jobs.ashbyhq.com/acme/9a15ed0b-1a0e-4c00-b7c8-8a0c4e8e9abc",
    )
    _apply_listing(opp, html)
    assert opp.remote is True


def test_ashby_to_html_workplace_type_wins_over_city_location():
    from src.engine import _apply_listing, _ashby_to_html

    html = _ashby_to_html(
        {
            "title": "Engineer",
            "employmentType": "FullTime",
            "workplaceType": "Remote",
            "locationName": "Austin, TX",
            "descriptionHtml": "<p>Build systems.</p>",
        }
    )
    opp = Opportunity(
        title="x",
        url="https://jobs.ashbyhq.com/acme/9a15ed0b-1a0e-4c00-b7c8-8a0c4e8e9abc",
    )
    _apply_listing(opp, html)
    assert opp.remote is True


def test_apply_listing_reads_workplace_from_listing():
    from src.engine import _apply_listing, _ashby_to_html, _lever_to_html, _workable_jobs_to_html

    hybrid = _ashby_to_html(
        {
            "title": "Engineer",
            "employmentType": "FullTime",
            "workplaceType": "Hybrid",
            "descriptionHtml": "<p>Build systems.</p>",
            "scrapeableCompensationSalarySummary": "$180K - $200K",
        }
    )
    ashby = Opportunity(
        title="x",
        url="https://jobs.ashbyhq.com/acme/9a15ed0b-1a0e-4c00-b7c8-8a0c4e8e9abc",
        remote=True,
    )
    _apply_listing(ashby, hybrid)
    assert ashby.remote is False
    assert ashby.pay_high == 200_000
    assert ashby.score() == 70.0

    lever = Opportunity(title="x", url="https://jobs.lever.co/acme/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", remote=True)
    _apply_listing(
        lever,
        _lever_to_html(
            {
                "text": "Engineer",
                "workplaceType": "onsite",
                "categories": {"commitment": "Full-time"},
                "description": "<p>Build systems. $160,000 - $180,000</p>",
            }
        ),
    )
    assert lever.remote is False
    assert lever.pay_low == 160_000
    assert lever.pay_high == 180_000

    workable = Opportunity(title="x", url="https://jobs.workable.com/view/abc", remote=True)
    _apply_listing(
        workable,
        _workable_jobs_to_html(
            {
                "title": "Engineer",
                "workplace": "hybrid",
                "employmentType": "Full-time",
                "description": "<p>$140,000 - $160,000</p>",
                "company": {"title": "Acme"},
            }
        ),
    )
    assert workable.remote is False
    assert workable.company == "Acme"
    assert workable.pay_low == 140_000
    assert workable.hours_per_week == 40

    body = Opportunity(title="Engineer", url="https://jobs.example/x", remote=True)
    _apply_listing(
        body,
        "<title>Engineer at Acme</title><p>This is a hybrid role in NYC. $120,000 - $140,000</p>",
    )
    assert body.remote is False
    assert body.pay_high == 140_000

    remote = Opportunity(title="x", url="https://jobs.lever.co/acme/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
    _apply_listing(
        remote,
        _lever_to_html(
            {
                "text": "Staff Software Engineer",
                "workplaceType": "remote",
                "categories": {"commitment": "Full-time"},
                "description": (
                    "<p>This role can be hybrid, or fully remote/virtually. $180,000 - $200,000</p>"
                ),
            }
        ),
    )
    assert remote.remote is True
    assert remote.pay_high == 200_000
    assert remote.score() == 100.0

    offered = Opportunity(title="Engineer", url="https://jobs.example/x")
    _apply_listing(
        offered,
        (
            "<title>Engineer at Acme</title>"
            "<p>This role can be hybrid, or fully remote/virtually. $180,000 - $200,000</p>"
        ),
    )
    assert offered.remote is True
    assert offered.pay_high == 200_000
    assert offered.score() == 100.0


def test_unspecified_workplace_uses_city_not_token():
    from src.engine import (
        _apply_listing,
        _apply_workplace,
        _ashby_to_html,
        _lever_to_html,
        _workday_to_html,
    )

    empty: dict = {}
    _apply_workplace(empty, "unspecified")
    assert "jobLocationType" not in empty
    _apply_workplace(empty, "unknown", "n/a", "none")
    assert "jobLocationType" not in empty
    city: dict = {}
    _apply_workplace(city, "unspecified", "San Francisco")
    assert city["jobLocationType"] == "ON_SITE"
    country: dict = {}
    _apply_workplace(country, "not specified", "United States")
    assert "jobLocationType" not in country
    typed: dict = {}
    _apply_workplace(typed, "Remote", "San Francisco")
    assert typed["jobLocationType"] == "TELECOMMUTE"
    hybrid: dict = {}
    _apply_workplace(hybrid, "hybrid", "Remote USA")
    assert hybrid["jobLocationType"] == "ON_SITE"

    office = Opportunity(
        title="x",
        url="https://jobs.lever.co/hive/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        remote=True,
    )
    _apply_listing(
        office,
        _lever_to_html(
            {
                "text": "Staff Machine Learning Engineer",
                "workplaceType": "unspecified",
                "categories": {"commitment": "Full-time", "location": "San Francisco"},
                "description": "<p>Remote-friendly team. $160,000 - $200,000</p>",
            }
        ),
    )
    assert office.remote is False
    assert office.pay_high == 200_000

    us = Opportunity(
        title="x",
        url="https://jobs.lever.co/acme/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        remote=True,
    )
    _apply_listing(
        us,
        _lever_to_html(
            {
                "text": "Engineer",
                "workplaceType": "unspecified",
                "categories": {"location": "United States"},
                "description": "<p>Build systems. $160,000 - $200,000</p>",
            }
        ),
    )
    assert us.remote is True

    remote = Opportunity(
        title="x",
        url="https://jobs.lever.co/spotify/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    )
    _apply_listing(
        remote,
        _lever_to_html(
            {
                "text": "Personalization Engineer",
                "workplaceType": "remote",
                "categories": {"location": "New York, NY"},
                "description": "<p>$180,000 - $200,000</p>",
            }
        ),
    )
    assert remote.remote is True

    ashby = Opportunity(
        title="x",
        url="https://jobs.ashbyhq.com/acme/9a15ed0b-1a0e-4c00-b7c8-8a0c4e8e9abc",
        remote=True,
    )
    _apply_listing(
        ashby,
        _ashby_to_html(
            {
                "title": "Engineer",
                "employmentType": "FullTime",
                "workplaceType": "unspecified",
                "locationName": "Austin, TX",
                "descriptionHtml": "<p>Remote-friendly team.</p>",
            }
        ),
    )
    assert ashby.remote is False

    wd_office = Opportunity(
        title="x",
        url="https://adobe.wd5.myworkdayjobs.com/en-US/external_experienced/job/x_R1",
        remote=True,
    )
    _apply_listing(
        wd_office,
        _workday_to_html(
            {
                "hiringOrganization": {"name": "Adobe"},
                "jobPostingInfo": {
                    "title": "Staff Machine Learning Engineer",
                    "timeType": "Full time",
                    "remoteType": "unspecified",
                    "location": "San Jose",
                    "jobDescription": "<p>Base Pay Range: $211,800 USD - $306,625 USD</p>",
                },
            }
        ),
    )
    assert wd_office.remote is False

    wd_us = Opportunity(
        title="x",
        url="https://sailpoint.wd1.myworkdayjobs.com/en-US/SailPoint/job/x_R1",
        remote=True,
    )
    _apply_listing(
        wd_us,
        _workday_to_html(
            {
                "hiringOrganization": {"name": "SailPoint"},
                "jobPostingInfo": {
                    "title": "Staff Machine Learning Engineer",
                    "timeType": "Full time",
                    "remoteType": "unspecified",
                    "location": "United States",
                    "jobDescription": "<p>Base Pay Range: $149,200 USD - $251,576 USD</p>",
                },
            }
        ),
    )
    assert wd_us.remote is True


def test_workplace_remote_or_hybrid_is_remote():
    from src.engine import _apply_listing, _greenhouse_to_html, _workplace_remote

    assert _workplace_remote("Remote or Hybrid") is True
    assert _workplace_remote("Hybrid / Remote") is True
    assert _workplace_remote("Distributed; Hybrid") is True
    assert _workplace_remote("hybrid") is False
    assert _workplace_remote("Flex") is False
    assert _workplace_remote("New York, NY (Hybrid)") is False
    assert _workplace_remote("Remote - United States") is True
    assert _workplace_remote("onsite only") is False
    assert _workplace_remote("Office-based") is False
    assert _workplace_remote("Office-first") is False
    assert _workplace_remote("office first") is False
    assert _workplace_remote("Remote, office-first") is True
    assert _workplace_remote("Site-based") is False
    assert _workplace_remote("Campus-Based") is False
    assert _workplace_remote("Field-Based") is False
    assert _workplace_remote("HQ-based") is False
    assert _workplace_remote("Laboratory-based") is False
    assert _workplace_remote("On-Campus") is False
    assert _workplace_remote("Remote, field-based") is True
    assert _workplace_remote("Remote, on-campus") is True

    html = _greenhouse_to_html(
        {
            "company_name": "Acme",
            "title": "Engineer",
            "location": {"name": "Remote or Hybrid"},
            "content": (
                "<p>Tier 1 (NYC/SF Bay Area/Seattle): $160,000 - $190,000</p>"
                "<p>Tier 3 (US - All Other): $140,000 - $170,000</p>"
            ),
        }
    )
    opp = Opportunity(
        title="Engineer",
        url="https://job-boards.greenhouse.io/acme/jobs/1",
        remote=True,
    )
    _apply_listing(opp, html)
    assert opp.remote is True
    assert opp.pay_low == 140_000
    assert opp.pay_high == 170_000
    assert opp.score() == 85.0

    city = Opportunity(
        title="Engineer",
        url="https://job-boards.greenhouse.io/acme/jobs/2",
        remote=True,
    )
    _apply_listing(
        city,
        _greenhouse_to_html(
            {
                "company_name": "Stripe",
                "title": "Account Executive, AI Sales",
                "location": {"name": "San Francisco, CA"},
                "content": "<p>Sell to AI companies. $180,000 - $200,000</p>",
            }
        ),
    )
    assert city.remote is False
    assert city.pay_high == 200_000

    country = Opportunity(
        title="Engineer",
        url="https://job-boards.greenhouse.io/acme/jobs/3",
        remote=True,
    )
    _apply_listing(
        country,
        _greenhouse_to_html(
            {
                "company_name": "Acme",
                "title": "Engineer",
                "location": {"name": "United States"},
                "content": "<p>Build systems. $180,000 - $200,000</p>",
            }
        ),
    )
    assert country.remote is True


def test_jsonld_city_location_is_office_when_type_missing():
    from src.engine import _apply_listing

    html = """
    <title>Engineer</title>
    <script type="application/ld+json">
    {"@type":"JobPosting","title":"Staff Machine Learning Engineer",
     "jobLocation":{"@type":"Place","address":{"addressLocality":"Mountain View","addressRegion":"California"}}}
    </script>
    <p>Build ML systems. $202,500 - $274,000</p>
    """
    opp = Opportunity(
        title="x",
        url="https://jobs.intuit.com/job/mountain-view/staff-machine-learning-engineer/27595/87369441616",
        remote=True,
    )
    _apply_listing(opp, html)
    assert opp.remote is False
    assert opp.pay_high == 274_000

    remote = Opportunity(title="x", url="https://jobs.example/x")
    _apply_listing(
        remote,
        """
        <script type="application/ld+json">
        {"@type":"JobPosting","title":"Engineer","jobLocationType":"TELECOMMUTE",
         "jobLocation":{"@type":"Place","address":{"addressLocality":"Austin","addressRegion":"TX"}}}
        </script>
        <p>Build systems. $180,000 - $200,000</p>
        """,
    )
    assert remote.remote is True

    country = Opportunity(title="x", url="https://jobs.example/y", remote=True)
    _apply_listing(
        country,
        """
        <script type="application/ld+json">
        {"@type":"JobPosting","title":"Engineer",
         "jobLocation":{"@type":"Place","address":{"addressCountry":"United States"}}}
        </script>
        <p>Build systems. $180,000 - $200,000</p>
        """,
    )
    assert country.remote is True


def test_ashby_to_html_foreign_summary_is_not_usd():
    from src.engine import _apply_listing, _ashby_to_html, _foreign_salary

    html = _ashby_to_html(
        {
            "title": "Engineer",
            "employmentType": "FullTime",
            "scrapeableCompensationSalarySummary": "€60,000 - €80,000",
        }
    )
    opp = Opportunity(
        title="x",
        url="https://jobs.ashbyhq.com/acme/9a15ed0b-1a0e-4c00-b7c8-8a0c4e8e9abc",
    )
    _apply_listing(opp, html)
    assert opp.pay_high is None
    assert _foreign_salary(html) is True


def test_ashby_posting_null_is_gone():
    from src.engine import _ashby_posting

    class _Resp:
        status_code = 200
        text = '{"data":{"jobPosting":null}}'

    class _Client:
        def __init__(self):
            self.url = None
            self.payload = None

        async def post(self, url, **kwargs):
            self.url = url
            self.payload = kwargs.get("json")
            return _Resp()

    client = _Client()
    jid = "23ce794a-4aa7-45a3-bcdc-cb156271057b"
    assert asyncio.run(_ashby_posting(client, "azx", jid)) is None
    assert client.url == "https://jobs.ashbyhq.com/api/non-user-graphql?op=ApiJobPosting"
    assert client.payload["variables"] == {
        "organizationHostedJobsPageName": "azx",
        "jobPostingId": jid,
    }


def test_ashby_posting_http_error_is_empty():
    from src.engine import _ashby_posting

    class _Resp:
        status_code = 500
        text = "nope"

    class _Client:
        async def post(self, _url, **_kwargs):
            return _Resp()

    assert asyncio.run(_ashby_posting(_Client(), "azx", "x")) == {}


def test_listing_text_ashby_graphql_null_is_gone(monkeypatch):
    engine = Engine()
    seen: list[tuple[str, str]] = []

    async def fake_ashby(_client, board: str, jid: str):
        seen.append((board, jid))
        return None

    async def fake_get(_client, _url: str):
        raise AssertionError("SPA HTML must not be fetched when GraphQL says gone")

    monkeypatch.setattr("src.engine._ashby_posting", fake_ashby)
    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    jid = "23ce794a-4aa7-45a3-bcdc-cb156271057b"
    html = asyncio.run(
        engine._listing_text(f"https://jobs.ashbyhq.com/azx/{jid}/application")
    )
    assert html is None
    assert seen == [("azx", jid)]


def test_listing_text_ashby_graphql_timeout_falls_back_to_html(monkeypatch):
    engine = Engine()
    seen: list[str] = []

    async def fake_ashby(_client, _board: str, _jid: str):
        return {}

    async def fake_get(_client, url: str):
        seen.append(url)
        return (
            "<html><script type='application/ld+json'>{"
            '"@type":"JobPosting","title":"ML Engineer",'
            '"hiringOrganization":{"name":"Quilter"},'
            '"baseSalary":{"currency":"USD","value":{"minValue":180000,"maxValue":200000,"unitText":"YEAR"}}'
            "}</script></html>"
        )

    monkeypatch.setattr("src.engine._ashby_posting", fake_ashby)
    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    url = "https://jobs.ashbyhq.com/quilter/9a15ed0b-1a0e-4c00-b7c8-8a0c4e8e9abc"
    html = asyncio.run(engine._listing_text(url))
    assert seen == [url]
    assert html and "JobPosting" in html


def test_listing_text_ashby_graphql_pay_from_posting(monkeypatch):
    engine = Engine()

    async def fake_ashby(_client, board: str, jid: str):
        assert (board, jid) == (
            "quilter",
            "9a15ed0b-1a0e-4c00-b7c8-8a0c4e8e9abc",
        )
        return {
            "title": "Machine Learning Engineer",
            "employmentType": "FullTime",
            "descriptionHtml": "<p>Build ML systems.</p>",
            "compensationTierSummary": "$180K - $200K • Offsite",
            "scrapeableCompensationSalarySummary": "$180K - $200K",
        }

    async def fake_get(_client, _url: str):
        raise AssertionError("SPA HTML must not be fetched when GraphQL has the posting")

    monkeypatch.setattr("src.engine._ashby_posting", fake_ashby)
    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    url = "https://jobs.ashbyhq.com/quilter/9a15ed0b-1a0e-4c00-b7c8-8a0c4e8e9abc"
    html = asyncio.run(engine._listing_text(url))
    from src.engine import _apply_listing

    opp = Opportunity(title="x", url=url)
    _apply_listing(opp, html)
    assert opp.pay_low == 180_000
    assert opp.pay_high == 200_000
    assert opp.hours_per_week == 40
    assert opp.score() == 100


def test_lever_eur_salary_range_is_foreign():
    from src.engine import _apply_listing, _foreign_salary, _lever_to_html

    html = _lever_to_html(
        {
            "text": "Senior ML Engineer (Europe-based/Remote)",
            "salaryRange": {
                "min": 60000,
                "max": 85000,
                "currency": "EUR",
                "interval": "per-year-salary",
            },
        }
    )
    opp = Opportunity(
        title="x",
        url="https://jobs.lever.co/swordhealth/50945411-2f43-421a-8bb8-86aa1de6d890",
    )
    _apply_listing(opp, html)
    assert opp.pay_high is None
    assert _foreign_salary(html) is True


def test_listing_text_greenhouse_api_404_is_gone(monkeypatch):
    engine = Engine()
    seen: list[str] = []

    async def fake_get(_client, url: str):
        seen.append(url)
        if "boards-api.greenhouse.io" in url:
            return None
        return "<title>Jobs at Reddit</title><p>Current openings</p>"

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    html = asyncio.run(
        engine._listing_text("https://job-boards.greenhouse.io/reddit/jobs/8084032")
    )
    assert seen == [
        "https://boards-api.greenhouse.io/v1/boards/reddit/jobs/8084032?pay_transparency=true"
    ]
    assert html is None


def test_greenhouse_hosted_ids_from_gh_jid_and_html():
    from src.engine import _ats_job_url, _greenhouse_hosted_ids, _is_index_page

    url = "https://www.samsara.com/company/careers/roles/7266357?gh_jid=7266357"
    assert _greenhouse_hosted_ids(url) == ("samsara", "7266357")
    assert _ats_job_url(url)
    assert not _is_index_page(
        {"url": url, "title": "Jobs at Samsara", "description": ""}
    )
    assert _greenhouse_hosted_ids(
        "https://www.samsara.com/company/careers/roles/7266357"
    ) is None
    assert _greenhouse_hosted_ids(
        "https://www.samsara.com/company/careers/roles/7266357",
        '<div id="greenhouse-job-7266357-auto"></div>',
    ) == ("samsara", "7266357")
    assert _greenhouse_hosted_ids(
        "https://www.samsara.com/company/careers/roles/7266357",
        "https://job-boards.greenhouse.io/samsara/jobs/7266357",
    ) == ("samsara", "7266357")
    assert (
        _greenhouse_hosted_ids(
            "https://www.example.com/company/careers/roles/7266357",
            "<title>Careers</title><p>No greenhouse embed</p>",
        )
        is None
    )
    assert _greenhouse_hosted_ids(
        "https://job-boards.greenhouse.io/reddit/jobs/6960831"
    ) is None
    vendor = "https://www.greenhouse.com/careers?gh_jid=1234567"
    embed = "https://job-boards.greenhouse.io/greenhouse/jobs/1234567"
    assert _greenhouse_hosted_ids(vendor, embed) is None
    assert not _ats_job_url(vendor)
    assert _is_index_page(
        {"url": vendor, "title": "Staff Software Engineer", "description": ""}
    )


def test_listing_text_hosted_greenhouse_reads_api(monkeypatch):
    engine = Engine()
    seen: list[str] = []

    async def fake_get(_client, url: str):
        seen.append(url)
        if "boards-api.greenhouse.io" in url:
            return json.dumps(
                {
                    "id": 7266357,
                    "title": "Staff Machine Learning Engineer - Edge AI",
                    "company_name": "Samsara",
                    "location": {"name": "Remote - US"},
                    "content": "<p>Build models.</p>",
                    "pay_input_ranges": [
                        {
                            "min_cents": 17864000,
                            "max_cents": 31900000,
                            "currency_type": "USD",
                        }
                    ],
                }
            )
        return "<title>SPA</title>"

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    url = "https://www.samsara.com/company/careers/roles/7266357?gh_jid=7266357"
    html = asyncio.run(engine._listing_text(url))
    assert seen == [
        "https://boards-api.greenhouse.io/v1/boards/samsara/jobs/7266357?pay_transparency=true"
    ]
    from src.engine import _apply_listing

    opp = Opportunity(title="x", url=url)
    _apply_listing(opp, html)
    assert opp.company == "Samsara"
    assert opp.title == "Staff Machine Learning Engineer - Edge AI"
    assert opp.remote is True
    assert opp.pay_low == 178_640
    assert opp.pay_high == 319_000
    assert opp.score() == 159.5


def test_listing_text_hosted_greenhouse_api_404_keeps_html(monkeypatch):
    engine = Engine()
    seen: list[str] = []

    async def fake_get(_client, url: str):
        seen.append(url)
        if "boards-api.greenhouse.io" in url:
            return None
        return (
            "<title>Staff Machine Learning Engineer - Edge AI - Remote - US</title>"
            "<p>$178,640 - $319,000</p>"
        )

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    url = "https://www.samsara.com/company/careers/roles/7266357?gh_jid=7266357"
    html = asyncio.run(engine._listing_text(url))
    assert any("boards-api.greenhouse.io" in u for u in seen)
    assert html and "$178,640" in html
    from src.engine import _apply_listing

    opp = Opportunity(title="x", url=url)
    _apply_listing(opp, html)
    assert opp.pay_high == 319_000
    assert opp.company is None


def test_listing_text_hosted_greenhouse_from_html_marker(monkeypatch):
    engine = Engine()
    seen: list[str] = []

    async def fake_get(_client, url: str):
        seen.append(url)
        if "boards-api.greenhouse.io" in url:
            return json.dumps(
                {
                    "id": 7266357,
                    "title": "Staff ML",
                    "company_name": "Samsara",
                    "location": {"name": "Remote - US"},
                    "content": "<p>x</p>",
                }
            )
        return '<div id="greenhouse-job-7266357-auto"></div>'

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    url = "https://www.samsara.com/company/careers/roles/7266357"
    html = asyncio.run(engine._listing_text(url))
    assert seen[0] == url
    assert seen[1] == (
        "https://boards-api.greenhouse.io/v1/boards/samsara/jobs/7266357"
        "?pay_transparency=true"
    )
    from src.engine import _apply_listing

    opp = Opportunity(title="x", url=url)
    _apply_listing(opp, html)
    assert opp.company == "Samsara"
    assert opp.remote is True


def test_listing_text_greenhouse_api_timeout_falls_back_to_html(monkeypatch):
    engine = Engine()
    seen: list[str] = []

    async def fake_get(_client, url: str):
        seen.append(url)
        if "boards-api.greenhouse.io" in url:
            return ""
        return "<title>Senior ML at Reddit</title><p>$180,000</p>"

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    html = asyncio.run(
        engine._listing_text("https://job-boards.greenhouse.io/reddit/jobs/6960831")
    )
    assert seen[0] == (
        "https://boards-api.greenhouse.io/v1/boards/reddit/jobs/6960831?pay_transparency=true"
    )
    assert seen[1] == "https://job-boards.greenhouse.io/reddit/jobs/6960831"
    assert html and "$180,000" in html


def test_enrich_drops_http_404_listings_keeps_empty_fetches():
    engine = Engine()

    async def page(url: str):
        if "gone" in url:
            return None
        if "thin" in url:
            return ""
        return "<title>Senior ML</title><p>$180,000 a year</p>"

    engine._listing_text = page
    priced = Opportunity(title="Priced", url="https://jobs.example/paid")
    ghost = Opportunity(title="Ghost", url="https://jobs.lever.co/gone/abc")
    thin = Opportunity(title="Thin", url="https://jobs.example/thin")
    opps = [priced, ghost, thin]
    asyncio.run(engine._enrich_pay(opps))
    assert [o.title for o in opps] == ["Priced", "Thin"]
    assert priced.pay_high == 180_000
    assert thin.pay_high is None


def test_greenhouse_api_url_from_job_board_link():
    from src.engine import _greenhouse_api_url, _lever_job_url, _normalize_url

    api = "https://boards-api.greenhouse.io/v1/boards/reddit/jobs/6960831?pay_transparency=true"
    assert _greenhouse_api_url("https://job-boards.greenhouse.io/reddit/jobs/6960831") == api
    assert _greenhouse_api_url(
        "https://boards.greenhouse.io/embed/job_app?for=reddit&token=6960831"
    ) == api
    assert _greenhouse_api_url(
        "https://job-boards.greenhouse.io/embed/job_app?token=6960831&for=reddit"
    ) == api
    assert _greenhouse_api_url(
        "https://job-boards.eu.greenhouse.io/jetbrains/jobs/4713663101"
    ) == "https://boards-api.greenhouse.io/v1/boards/jetbrains/jobs/4713663101?pay_transparency=true"
    assert _greenhouse_api_url(
        "https://boards.eu.greenhouse.io/jetbrains/jobs/4713663101"
    ) == "https://boards-api.greenhouse.io/v1/boards/jetbrains/jobs/4713663101?pay_transparency=true"
    assert _greenhouse_api_url("https://jobs.lever.co/lyrahealth/abc") is None
    assert _lever_job_url(
        "https://boards.greenhouse.io/embed/job_app?for=reddit&token=6960831"
    ) == "https://job-boards.greenhouse.io/reddit/jobs/6960831"
    assert _normalize_url(
        "https://boards.greenhouse.io/embed/job_app?for=reddit&token=6960831"
    ) == _normalize_url("https://job-boards.greenhouse.io/reddit/jobs/6960831")


def test_greenhouse_api_html_fills_company_and_pay_range():
    from src.engine import _apply_listing, _greenhouse_to_html

    html = _greenhouse_to_html(
        {
            "company_name": "Reddit",
            "title": "Senior Machine Learning Engineer",
            "location": {"name": "Remote - United States"},
            "content": (
                "&lt;div class=&quot;pay-range&quot;&gt;"
                "&lt;span&gt;$216,700&lt;/span&gt;&lt;span&gt;&amp;mdash;&lt;/span&gt;"
                "&lt;span&gt;$303,400 USD&lt;/span&gt;&lt;/div&gt;"
            ),
        }
    )
    opp = Opportunity(
        title="Senior Machine Learning Engineer, ML Efficiency",
        url="https://job-boards.greenhouse.io/reddit/jobs/6960831",
    )
    _apply_listing(opp, html)
    assert opp.company == "Reddit"
    assert opp.title == "Senior Machine Learning Engineer"
    assert opp.pay_low == 216_700
    assert opp.pay_high == 303_400


def test_greenhouse_pay_transparency_fills_json_ld_without_content_dollars():
    from src.engine import _apply_listing, _foreign_salary, _greenhouse_to_html

    html = _greenhouse_to_html(
        {
            "company_name": "Reddit",
            "title": "Senior ML",
            "content": "<p>Apply now. No figures in the body.</p>",
            "pay_input_ranges": [
                {
                    "min_cents": 21670000,
                    "max_cents": 30340000,
                    "currency_type": "USD",
                    "title": "The base salary range for this position is:",
                },
                {
                    "min_cents": 10000000,
                    "max_cents": 12000000,
                    "currency_type": "EUR",
                },
            ],
        }
    )
    opp = Opportunity(title="x", url="https://job-boards.greenhouse.io/reddit/jobs/6960831")
    _apply_listing(opp, html)
    assert opp.pay_low == 216_700
    assert opp.pay_high == 303_400
    assert opp.hours_per_week is None

    eur_only = _greenhouse_to_html(
        {
            "company_name": "Acme",
            "title": "Engineer",
            "content": "<p>Apply now.</p>",
            "pay_input_ranges": [
                {"min_cents": 12000000, "max_cents": 18000000, "currency_type": "EUR"}
            ],
        }
    )
    skipped = Opportunity(title="Engineer", url="https://job-boards.greenhouse.io/acme/jobs/1")
    _apply_listing(skipped, eur_only)
    assert skipped.pay_high is None
    assert _foreign_salary(eur_only) is True

    cad = Opportunity(
        title="x",
        url="https://job-boards.greenhouse.io/samsara/jobs/7746586",
        remote=True,
    )
    cad_html = _greenhouse_to_html(
        {
            "company_name": "Samsara",
            "title": "Lead Machine Learning Engineer - ML Infrastructure",
            "location": {"name": "Remote - Canada"},
            "content": "<p>Annual Base Salary$196,000—$269,500 CADTotal Rewards</p>",
            "pay_input_ranges": [
                {
                    "min_cents": 19600000,
                    "max_cents": 26950000,
                    "currency_type": "CAD",
                    "title": "Annual Base Salary",
                }
            ],
        }
    )
    assert _foreign_salary(cad_html) is True
    assert _apply_listing(cad, cad_html) is False
    assert cad.pay_high is None
    assert cad.remote is True


def test_greenhouse_metadata_scheduled_hours_and_time_type():
    from src.engine import _apply_listing, _greenhouse_to_html

    html = _greenhouse_to_html(
        {
            "company_name": "Reddit",
            "title": "Senior Machine Learning Engineer",
            "content": "<p>Apply now. No hours in the body.</p>",
            "pay_input_ranges": [
                {
                    "min_cents": 21670000,
                    "max_cents": 30340000,
                    "currency_type": "USD",
                }
            ],
            "metadata": [
                {"name": "Time Type", "value": "Full time", "value_type": "single_select"},
                {"name": "Scheduled Weekly Hours", "value": "40.0", "value_type": "number"},
                {"name": "Worker Sub-Type", "value": "Regular", "value_type": "single_select"},
            ],
        }
    )
    opp = Opportunity(title="x", url="https://job-boards.greenhouse.io/reddit/jobs/6960831")
    _apply_listing(opp, html)
    assert opp.pay_low == 216_700
    assert opp.pay_high == 303_400
    assert opp.hours_per_week == 40
    assert opp.rate_is_imputed is False
    assert opp.score() == 151.7


def test_apply_listing_prefers_remote_geo_band_over_json_ld():
    from src.engine import _apply_listing

    html = """
    <script type="application/ld+json">
    {"@type":"JobPosting","title":"Engineer",
     "baseSalary":{"currency":"USD","value":{"minValue":160000,"maxValue":190000,"unitText":"YEAR"}}}
    </script>
    <p>Tier 1 (NYC/SF Bay Area/Seattle): $160,000 - $190,000</p>
    <p>Tier 2 (DC Metro/Austin/Boston/Los Angeles): $150,000 - $180,000</p>
    <p>Tier 3 (US - All Other): $140,000 - $170,000</p>
    """
    remote = Opportunity(title="x", url="https://jobs.example/x", remote=True)
    _apply_listing(remote, html)
    assert remote.pay_low == 140_000
    assert remote.pay_high == 170_000

    office = Opportunity(title="x", url="https://jobs.example/x", remote=True)
    office_html = html.replace(
        "<p>Tier 1",
        "<p>This is a hybrid role.</p><p>Tier 1",
    )
    _apply_listing(office, office_html)
    assert office.remote is False
    assert office.pay_low == 160_000
    assert office.pay_high == 190_000


def test_greenhouse_geo_bands_use_all_other_when_remote():
    from src.engine import _apply_listing, _greenhouse_to_html

    html = _greenhouse_to_html(
        {
            "company_name": "Signifyd",
            "title": "Senior Machine Learning Engineer",
            "location": {"name": "Remote, USA"},
            "content": (
                "<p>Tier 1 (NYC/SF Bay Area/Seattle): $160,000 - $190,000</p>"
                "<p>Tier 2 (DC Metro/Austin/Boston/Los Angeles): $150,000 - $180,000</p>"
                "<p>Tier 3 (US - All Other): $140,000 - $170,000</p>"
            ),
        }
    )
    opp = Opportunity(
        title="Senior Machine Learning Engineer",
        url="https://job-boards.greenhouse.io/signifyd/jobs/1",
        remote=True,
    )
    _apply_listing(opp, html)
    assert opp.company == "Signifyd"
    assert opp.pay_low == 140_000
    assert opp.pay_high == 170_000
    assert opp.score() == 85.0


def test_apply_listing_guesses_hours_when_json_ld_already_has_pay():
    from src.engine import _apply_listing

    html = """
    <script type="application/ld+json">
    {"@type":"JobPosting","title":"Engineer",
     "baseSalary":{"currency":"USD","value":{"minValue":160000,"maxValue":190000,"unitText":"YEAR"}}}
    </script>
    <p>This is a full-time position.</p>
    """
    opp = Opportunity(title="x", url="https://jobs.example/x")
    _apply_listing(opp, html)
    assert opp.pay_low == 160_000
    assert opp.pay_high == 190_000
    assert opp.hours_per_week == 40


def test_workable_jobs_api_url_from_view_link():
    from src.engine import _workable_jobs_api_url

    assert _workable_jobs_api_url(
        "https://jobs.workable.com/view/3wwPqWr4G8nzLWnxfEAKur/remote-senior-engineer-ai-ml"
    ) == "https://jobs.workable.com/api/v1/jobs/3wwPqWr4G8nzLWnxfEAKur"
    assert _workable_jobs_api_url("https://apply.workable.com/a2z-sync/j/C95E51CDDA") is None


def test_workable_jobs_api_html_fills_company_and_pay_range():
    from src.engine import _apply_listing, _workable_jobs_to_html

    html = _workable_jobs_to_html(
        {
            "title": "Senior Engineer, AI/ML",
            "company": {"title": "A2Z Sync"},
            "employmentType": "Full-time",
            "description": "<p>Build agents.</p>",
            "requirementsSection": (
                "<p>The expected salary range for this role is "
                "<strong>$160,000 to $190,000 annually</strong>.</p>"
            ),
        }
    )
    opp = Opportunity(
        title="Senior Engineer, AI/ML | A2Z Sync | Jobs By Workable",
        url="https://jobs.workable.com/view/3wwPqWr4G8nzLWnxfEAKur/x",
    )
    _apply_listing(opp, html)
    assert opp.company == "A2Z Sync"
    assert opp.title == "Senior Engineer, AI/ML"
    assert opp.pay_low == 160_000
    assert opp.pay_high == 190_000
    assert opp.hours_per_week == 40


def test_listing_text_prefers_workable_jobs_api_over_spa_shell(monkeypatch):
    engine = Engine()
    seen: list[str] = []

    async def fake_get(_client, url: str) -> str:
        seen.append(url)
        if "/api/v1/jobs/" in url:
            return json.dumps(
                {
                    "title": "Senior Machine Learning Engineer",
                    "company": {"title": "Canopy"},
                    "requirementsSection": "<p>Base Salary: $126,000 - $180,000</p>",
                }
            )
        return "<title>Senior Machine Learning Engineer | Canopy | Jobs By Workable</title>"

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    html = asyncio.run(
        engine._listing_text(
            "https://jobs.workable.com/view/7mMjfHgS93LyPeHLK2XeMV/remote-senior-machine-learning-engineer"
        )
    )
    assert seen[0] == "https://jobs.workable.com/api/v1/jobs/7mMjfHgS93LyPeHLK2XeMV"
    from src.engine import _apply_listing

    opp = Opportunity(
        title="Senior Machine Learning Engineer | Canopy | Jobs By Workable",
        url="https://jobs.workable.com/view/7mMjfHgS93LyPeHLK2XeMV/x",
    )
    _apply_listing(opp, html)
    assert opp.company == "Canopy"
    assert opp.pay_low == 126_000
    assert opp.pay_high == 180_000


def test_listing_text_prefers_greenhouse_api_over_board_shell(monkeypatch):
    engine = Engine()
    seen: list[str] = []

    async def fake_get(_client, url: str) -> str:
        seen.append(url)
        if "boards-api.greenhouse.io" in url:
            return json.dumps(
                {
                    "company_name": "Reddit",
                    "title": "Senior ML",
                    "content": "$180,000",
                    "location": {"name": "Remote - United States"},
                }
            )
        return "<title>Jobs at Reddit</title>"

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    html = asyncio.run(
        engine._listing_text("https://job-boards.greenhouse.io/reddit/jobs/6960831")
    )
    assert seen[0] == (
        "https://boards-api.greenhouse.io/v1/boards/reddit/jobs/6960831?pay_transparency=true"
    )
    from src.engine import _apply_listing

    opp = Opportunity(title="x", url="https://job-boards.greenhouse.io/reddit/jobs/6960831")
    _apply_listing(opp, html)
    assert opp.company == "Reddit"
    assert opp.pay_high == 180_000


def test_listing_text_reads_greenhouse_embed_via_api(monkeypatch):
    engine = Engine()
    seen: list[str] = []

    async def fake_get(_client, url: str) -> str:
        seen.append(url)
        if "boards-api.greenhouse.io" in url:
            return json.dumps(
                {
                    "company_name": "Reddit",
                    "title": "Senior ML",
                    "content": "$180,000",
                    "location": {"name": "Remote - United States"},
                }
            )
        return "<title>Jobs at Reddit</title>"

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    html = asyncio.run(
        engine._listing_text(
            "https://boards.greenhouse.io/embed/job_app?for=reddit&token=6960831"
        )
    )
    assert seen[0] == (
        "https://boards-api.greenhouse.io/v1/boards/reddit/jobs/6960831?pay_transparency=true"
    )
    from src.engine import _apply_listing

    opp = Opportunity(
        title="x",
        url="https://boards.greenhouse.io/embed/job_app?for=reddit&token=6960831",
    )
    _apply_listing(opp, html)
    assert opp.company == "Reddit"
    assert opp.pay_high == 180_000
    assert opp.remote is True


def test_smartrecruiters_api_url_from_job_link():
    from src.engine import (
        _is_index_page,
        _lever_job_url,
        _smartrecruiters_api_url,
    )

    api = "https://api.smartrecruiters.com/v1/companies/Socotec/postings/744000141322430"
    assert (
        _smartrecruiters_api_url(
            "https://jobs.smartrecruiters.com/Socotec/744000141322430-applied-ai-engineer"
        )
        == api
    )
    assert _lever_job_url(
        "https://jobs.smartrecruiters.com/Socotec/744000141322430-applied-ai-engineer"
    ) == "https://jobs.smartrecruiters.com/Socotec/744000141322430"
    assert _is_index_page(
        {"url": "https://jobs.smartrecruiters.com/Socotec", "title": "SOCOTEC", "description": ""}
    )
    assert not _is_index_page(
        {
            "url": "https://jobs.smartrecruiters.com/Socotec/744000141322430",
            "title": "Applied AI Engineer",
            "description": "",
        }
    )
    assert _smartrecruiters_api_url("https://jobs.lever.co/acme/x") is None


def test_smartrecruiters_to_html_fills_company_pay_and_remote():
    from src.engine import _apply_listing, _smartrecruiters_to_html

    html = _smartrecruiters_to_html(
        {
            "name": "Applied AI Engineer",
            "company": {"name": "SOCOTEC"},
            "typeOfEmployment": {"id": "permanent", "label": "Full-time"},
            "location": {
                "city": "New York",
                "remote": False,
                "hybrid": False,
                "fullLocation": "New York, United States",
            },
            "jobAd": {
                "sections": {
                    "additionalInformation": {"text": "<p>Salary: $157-200k</p>"},
                }
            },
        }
    )
    opp = Opportunity(
        title="x",
        url="https://jobs.smartrecruiters.com/Socotec/744000141322430",
        remote=True,
    )
    _apply_listing(opp, html)
    assert opp.company == "SOCOTEC"
    assert opp.title == "Applied AI Engineer"
    assert opp.pay_low == 157_000
    assert opp.pay_high == 200_000
    assert opp.remote is False
    assert opp.hours_per_week == 40
    assert opp.score() == 70.0

    remote = Opportunity(title="x", url="https://jobs.smartrecruiters.com/mirantis/1")
    _apply_listing(
        remote,
        _smartrecruiters_to_html(
            {
                "name": "Senior Software Engineer (Golang)",
                "company": {"name": "Mirantis"},
                "typeOfEmployment": {"label": "Full-time"},
                "location": {"city": "Remote", "remote": True, "hybrid": False},
                "jobAd": {"sections": {"jobDescription": {"text": "<p>Go systems.</p>"}}},
            }
        ),
    )
    assert remote.company == "Mirantis"
    assert remote.remote is True
    assert remote.pay_high is None


def test_smartrecruiters_api_compensation_ranks_usd_and_drops_foreign():
    from src.engine import _apply_listing, _foreign_salary, _smartrecruiters_to_html

    usd = Opportunity(
        title="x",
        url="https://jobs.smartrecruiters.com/ServiceNow/744000147354611",
        remote=True,
    )
    _apply_listing(
        usd,
        _smartrecruiters_to_html(
            {
                "name": "Enterprise Account Exec (Armis/Veza)",
                "company": {"name": "ServiceNow"},
                "typeOfEmployment": {"label": "Full-time"},
                "location": {
                    "city": "Raleigh",
                    "remote": True,
                    "hybrid": False,
                    "fullLocation": "Raleigh, NC, United States",
                },
                "compensation": {
                    "min": 114400,
                    "max": 165000,
                    "currency": "USD",
                    "period": "YEARLY",
                },
                "jobAd": {"sections": {"jobDescription": {"text": "<p>Sell software.</p>"}}},
            }
        ),
    )
    assert usd.company == "ServiceNow"
    assert usd.remote is True
    assert usd.pay_low == 114_400
    assert usd.pay_high == 165_000
    assert usd.hours_per_week == 40

    gbp_html = _smartrecruiters_to_html(
        {
            "name": "Lead Data Scientist - Pricing",
            "company": {"name": "Wise"},
            "typeOfEmployment": {"label": "Full-time"},
            "location": {"city": "London", "remote": False, "hybrid": False},
            "compensation": {
                "min": 90500,
                "max": 127000,
                "currency": "GBP",
                "period": "YEARLY",
            },
            "jobAd": {"sections": {"jobDescription": {"text": "<p>Price FX.</p>"}}},
        }
    )
    assert _foreign_salary(gbp_html) is True
    gbp = Opportunity(
        title="x",
        url="https://jobs.smartrecruiters.com/Wise/744000147450869",
    )
    assert _apply_listing(gbp, gbp_html) is False
    assert gbp.pay_high is None

    eur_html = _smartrecruiters_to_html(
        {
            "name": "Senior Operations Analyst",
            "company": {"name": "Wise"},
            "typeOfEmployment": {"label": "Full-time"},
            "location": {"city": "Tallinn", "remote": False},
            "compensation": {
                "min": 3500,
                "max": 5000,
                "currency": "EUR",
                "period": "MONTHLY",
            },
            "jobAd": {"sections": {"jobDescription": {"text": "<p>Payments ops.</p>"}}},
        }
    )
    assert _foreign_salary(eur_html) is True


def test_listing_text_reads_smartrecruiters_api(monkeypatch):
    engine = Engine()
    seen: list[str] = []

    async def fake_get(_client, url: str) -> str:
        seen.append(url)
        if "api.smartrecruiters.com" in url:
            return json.dumps(
                {
                    "name": "Applied AI Engineer",
                    "company": {"name": "SOCOTEC"},
                    "typeOfEmployment": {"label": "Full-time"},
                    "location": {"remote": False, "hybrid": False, "city": "New York"},
                    "jobAd": {
                        "sections": {
                            "additionalInformation": {"text": "<p>Salary: $157-200k</p>"}
                        }
                    },
                }
            )
        return "<title>Jobs at SOCOTEC</title>"

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    html = asyncio.run(
        engine._listing_text(
            "https://jobs.smartrecruiters.com/Socotec/744000141322430-applied-ai-engineer"
        )
    )
    assert seen == [
        "https://api.smartrecruiters.com/v1/companies/Socotec/postings/744000141322430"
    ]
    from src.engine import _apply_listing

    opp = Opportunity(title="x", url="https://jobs.smartrecruiters.com/Socotec/744000141322430")
    _apply_listing(opp, html)
    assert opp.company == "SOCOTEC"
    assert opp.pay_high == 200_000
    assert opp.remote is False


def test_workday_api_url_from_job_link():
    from src.engine import _is_index_page, _workday_api_url

    assert _workday_api_url(
        "https://workday.wd5.myworkdayjobs.com/en-US/Workday/job/Machine-Learning-Engineer_JR-0106147"
    ) == (
        "https://workday.wd5.myworkdayjobs.com/wday/cxs/workday/Workday/job/"
        "Machine-Learning-Engineer_JR-0106147"
    )
    assert _workday_api_url(
        "https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite/job/"
        "US-CA-Santa-Clara/Machine-Learning-Engineer--AI-Safety_JR2021784-1"
    ) == (
        "https://nvidia.wd5.myworkdayjobs.com/wday/cxs/nvidia/NVIDIAExternalCareerSite/job/"
        "Machine-Learning-Engineer--AI-Safety_JR2021784-1"
    )
    assert _is_index_page(
        {
            "url": "https://workday.wd5.myworkdayjobs.com/en-US/Workday",
            "title": "Workday Careers",
            "description": "",
        }
    )
    assert not _is_index_page(
        {
            "url": "https://workday.wd5.myworkdayjobs.com/en-US/Workday/job/Machine-Learning-Engineer_JR-0106147",
            "title": "Machine Learning Engineer III",
            "description": "",
        }
    )
    assert _workday_api_url("https://jobs.lever.co/acme/x") is None


def test_workday_to_html_fills_company_pay_and_flex():
    from src.engine import _apply_listing, _workday_to_html

    html = _workday_to_html(
        {
            "hiringOrganization": {"name": "Workday, Inc."},
            "jobPostingInfo": {
                "title": "Machine Learning Engineer III",
                "timeType": "Full Time",
                "remoteType": "Flex",
                "location": "USA, CA, Pleasanton",
                "jobDescription": "<p>Base Pay Range: $160,000 USD - $240,000 USD</p>",
            },
        }
    )
    opp = Opportunity(
        title="x",
        url="https://workday.wd5.myworkdayjobs.com/en-US/Workday/job/x_JR-1",
        remote=True,
    )
    _apply_listing(opp, html)
    assert opp.company == "Workday, Inc."
    assert opp.title == "Machine Learning Engineer III"
    assert opp.pay_low == 160_000
    assert opp.pay_high == 240_000
    assert opp.remote is False
    assert opp.hours_per_week == 40
    assert opp.score() == 84.0

    office = Opportunity(
        title="x",
        url="https://adobe.wd5.myworkdayjobs.com/en-US/external_experienced/job/x_R1",
        remote=True,
    )
    _apply_listing(
        office,
        _workday_to_html(
            {
                "hiringOrganization": {"name": "ADUS-Adobe Inc."},
                "jobPostingInfo": {
                    "title": "Staff Machine Learning Engineer",
                    "timeType": "Full time",
                    "location": "San Jose",
                    "jobDescription": "<p>Base Pay Range: $211,800 USD - $306,625 USD</p>",
                },
            }
        ),
    )
    assert office.remote is False
    assert office.pay_high == 306_625
    assert office.score() == 107.31875

    country = Opportunity(
        title="x",
        url="https://sailpoint.wd1.myworkdayjobs.com/en-US/SailPoint/job/x_R1",
        remote=True,
    )
    _apply_listing(
        country,
        _workday_to_html(
            {
                "hiringOrganization": {"name": "SailPoint Technologies, Inc."},
                "jobPostingInfo": {
                    "title": "Staff Machine Learning Engineer",
                    "timeType": "Full time",
                    "location": "United States",
                    "jobDescription": "<p>Base Pay Range: $149,200 USD - $251,576 USD</p>",
                },
            }
        ),
    )
    assert country.remote is True
    assert country.pay_high == 251_576


def test_workday_additional_us_remote_is_remote():
    from src.engine import _apply_listing, _workday_to_html

    remote = Opportunity(
        title="x",
        url="https://capitalone.wd12.myworkdayjobs.com/en-US/Capital_One/job/x_R1",
        remote=True,
    )
    _apply_listing(
        remote,
        _workday_to_html(
            {
                "hiringOrganization": {"name": "Capital One"},
                "jobPostingInfo": {
                    "title": "Sr. Staff Machine Learning Engineer (Remote-Eligible)",
                    "timeType": "Full time",
                    "location": "McLean, VA",
                    "additionalLocations": ["US Remote"],
                    "jobDescription": (
                        "<p>Capital One is open to hiring a Remote Employee.</p>"
                        "<p>Remote (Regardless of Location): $286,200 - $326,700</p>"
                    ),
                },
            }
        ),
    )
    assert remote.remote is True
    assert remote.pay_low == 286_200
    assert remote.pay_high == 326_700

    flex = Opportunity(
        title="x",
        url="https://workday.wd5.myworkdayjobs.com/en-US/Workday/job/x_JR-1",
        remote=True,
    )
    _apply_listing(
        flex,
        _workday_to_html(
            {
                "hiringOrganization": {"name": "Workday, Inc."},
                "jobPostingInfo": {
                    "title": "Machine Learning Engineer III",
                    "timeType": "Full Time",
                    "remoteType": "Flex",
                    "location": "USA, CA, Pleasanton",
                    "additionalLocations": ["USA, CA, San Francisco"],
                    "jobDescription": "<p>Base Pay Range: $160,000 USD - $240,000 USD</p>",
                },
            }
        ),
    )
    assert flex.remote is False


def test_listing_text_reads_workday_cxs(monkeypatch):
    engine = Engine()
    seen: list[str] = []

    async def fake_get(_client, url: str):
        seen.append(url)
        if "/wday/cxs/" in url:
            return json.dumps(
                {
                    "hiringOrganization": {"name": "Workday, Inc."},
                    "jobPostingInfo": {
                        "title": "Machine Learning Engineer III",
                        "timeType": "Full Time",
                        "remoteType": "Remote",
                        "jobDescription": "<p>Base Pay Range: $160,000 USD - $240,000 USD</p>",
                    },
                }
            )
        return "<title>Jobs at Workday</title>"

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    html = asyncio.run(
        engine._listing_text(
            "https://workday.wd5.myworkdayjobs.com/en-US/Workday/job/Machine-Learning-Engineer_JR-0106147"
        )
    )
    assert seen == [
        "https://workday.wd5.myworkdayjobs.com/wday/cxs/workday/Workday/job/"
        "Machine-Learning-Engineer_JR-0106147"
    ]
    from src.engine import _apply_listing

    opp = Opportunity(title="x", url="https://workday.wd5.myworkdayjobs.com/en-US/Workday/job/x")
    _apply_listing(opp, html)
    assert opp.company == "Workday, Inc."
    assert opp.pay_high == 240_000
    assert opp.remote is True
    assert opp.score() == 120.0


def test_listing_text_workday_cxs_404_falls_back_to_html(monkeypatch):
    engine = Engine()
    seen: list[str] = []

    async def fake_get(_client, url: str):
        seen.append(url)
        if "/wday/cxs/" in url:
            return None
        return (
            "<title>Engineer at Motorola</title>"
            "<p>This is a full-time remote role. $180,000 - $200,000</p>"
        )

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    html = asyncio.run(
        engine._listing_text(
            "https://motorolasolutions.wd5.myworkdayjobs.com/en-US/Careers/job/Machine-Learning-Engineer_R64440"
        )
    )
    assert seen[0].startswith("https://motorolasolutions.wd5.myworkdayjobs.com/wday/cxs/")
    assert seen[1] == (
        "https://motorolasolutions.wd5.myworkdayjobs.com/en-US/Careers/job/"
        "Machine-Learning-Engineer_R64440"
    )
    assert html and "$180,000" in html


def test_listing_text_workday_empty_spa_is_gone(monkeypatch):
    engine = Engine()
    seen: list[str] = []

    async def fake_get(_client, url: str):
        seen.append(url)
        if "/wday/cxs/" in url:
            return ""
        return "<!DOCTYPE html><html><head><title></title></head><body></body></html>"

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    html = asyncio.run(
        engine._listing_text(
            "https://shipt.wd1.myworkdayjobs.com/en-US/Shipt_External/job/"
            "Staff-Machine-Learning-Engineer_R4230"
        )
    )
    assert seen[0].endswith("/wday/cxs/shipt/Shipt_External/job/Staff-Machine-Learning-Engineer_R4230")
    assert any("Shipt_External/job/Staff-Machine-Learning-Engineer_R4230" in u and "/wday/cxs/" not in u for u in seen)
    assert html is None


def test_icims_iframe_url_from_job_link():
    from src.engine import _icims_iframe_url, _is_index_page, _lever_job_url

    pretty = (
        "https://uscareers-yelp.icims.com/jobs/13815/"
        "senior-machine-learning-engineer---content/job"
    )
    assert (
        _icims_iframe_url(pretty)
        == "https://uscareers-yelp.icims.com/jobs/13815/job?in_iframe=1"
    )
    assert _lever_job_url(pretty) == "https://uscareers-yelp.icims.com/jobs/13815/job"
    assert not _is_index_page(
        {"url": pretty, "title": "Careers at Yelp | Yelp Jobs", "description": ""}
    )
    assert _is_index_page(
        {
            "url": "https://careers-mci.icims.com/jobs/intro",
            "title": "Careers Center | Welcome",
            "description": "",
        }
    )
    assert _icims_iframe_url("https://jobs.lever.co/acme/x") is None


def test_listing_text_reads_icims_iframe(monkeypatch):
    engine = Engine()
    seen: list[str] = []
    iframe_html = (
        "<title>Senior ML Engineer</title>"
        '<script type="application/ld+json">'
        '{"@type":"JobPosting","title":"Senior ML Engineer",'
        '"hiringOrganization":{"name":"Yelp, Inc"},'
        '"jobLocationType":"TELECOMMUTE"}'
        "</script>"
        "<p>Compensation range for this role to be between $112,000 and $269,000.</p>"
    )

    async def fake_get(_client, url: str):
        seen.append(url)
        if "in_iframe=1" in url:
            return iframe_html
        return "<title>Careers at Yelp | Yelp Jobs</title>"

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    html = asyncio.run(
        engine._listing_text(
            "https://uscareers-yelp.icims.com/jobs/13815/senior-machine-learning-engineer/job"
        )
    )
    assert seen == ["https://uscareers-yelp.icims.com/jobs/13815/job?in_iframe=1"]
    from src.engine import _apply_listing

    opp = Opportunity(title="x", url="https://uscareers-yelp.icims.com/jobs/13815/job")
    _apply_listing(opp, html)
    assert opp.company == "Yelp, Inc"
    assert opp.pay_high == 269_000
    assert opp.remote is True


def test_listing_text_icims_410_is_gone(monkeypatch):
    engine = Engine()
    seen: list[str] = []

    async def fake_get(_client, url: str):
        seen.append(url)
        if "in_iframe=1" in url:
            return None
        return "<title>Careers at Acme | Acme Jobs</title><p>Search jobs</p>"

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    html = asyncio.run(
        engine._listing_text("https://careers-americas.icims.com/jobs/26849/principal-ml/job")
    )
    assert seen == ["https://careers-americas.icims.com/jobs/26849/job?in_iframe=1"]
    assert html is None


def test_jobvite_job_url_from_listing_link():
    from src.engine import _is_index_page, _jobvite_job_url, _lever_job_url

    url = "https://jobs.jobvite.com/brahma/job/ovU4zfwM"
    assert _jobvite_job_url(url) == url
    assert _lever_job_url(url) == url
    assert not _is_index_page(
        {"url": url, "title": "BRAHMA Careers - Machine Learning Engineer", "description": ""}
    )
    assert _is_index_page(
        {"url": "https://jobs.jobvite.com/brahma/jobs", "title": "BRAHMA Careers", "description": ""}
    )
    assert _is_index_page(
        {
            "url": "http://careers.jobvite.com/samtec/preview.htm",
            "title": "Samtec Careers - Jobvite",
            "description": "",
        }
    )
    assert _jobvite_job_url("https://jobs.lever.co/acme/x") is None


def test_listing_text_reads_jobvite_html(monkeypatch):
    engine = Engine()
    seen: list[str] = []
    html = (
        "<title>BRAHMA Careers - Machine Learning Engineer</title>"
        '<script type="application/ld+json">'
        '{"@type":"JobPosting","title":"Machine Learning Engineer",'
        '"hiringOrganization":{"name":"BRAHMA"},'
        '"jobLocationType":"TELECOMMUTE"}'
        "</script>"
        "<p>Remote, London, United Kingdom</p>"
    )

    async def fake_get(_client, url: str):
        seen.append(url)
        return html

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    text = asyncio.run(engine._listing_text("https://jobs.jobvite.com/brahma/job/ovU4zfwM"))
    assert seen == ["https://jobs.jobvite.com/brahma/job/ovU4zfwM"]
    from src.engine import _apply_listing

    opp = Opportunity(title="x", url="https://jobs.jobvite.com/brahma/job/ovU4zfwM")
    _apply_listing(opp, text)
    assert opp.company == "BRAHMA"
    assert opp.remote is True


def test_listing_text_jobvite_gone_html_is_gone(monkeypatch):
    engine = Engine()
    seen: list[str] = []

    async def fake_get(_client, url: str):
        seen.append(url)
        return "<title>FirstBank Careers</title><p>The job listing no longer exists.</p>"

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    html = asyncio.run(
        engine._listing_text("https://jobs.jobvite.com/firstbank/job/oqLpAfwU")
    )
    assert seen == ["https://jobs.jobvite.com/firstbank/job/oqLpAfwU"]
    assert html is None


def test_teamtailor_and_personio_job_urls_are_not_boards():
    from src.engine import _is_index_page

    assert _is_index_page(
        {
            "url": "https://rebtel.teamtailor.com/jobs",
            "title": "Current job openings - Rebtel",
            "description": "",
        }
    )
    assert not _is_index_page(
        {
            "url": "https://rebtel.teamtailor.com/jobs/7805704-senior-machine-learning-engineer",
            "title": "Senior Machine Learning Engineer - Rebtel",
            "description": "",
        }
    )
    assert _is_index_page(
        {
            "url": "https://mse-solutions.jobs.personio.com/?language=en",
            "title": "Jobs at mSE Solutions",
            "description": "",
        }
    )
    assert not _is_index_page(
        {
            "url": "https://swisspost.jobs.personio.com/job/2685789?language=en",
            "title": "Senior Machine Learning Engineer | Jobs at Swiss Post | IT Campus",
            "description": "",
        }
    )
    assert not _is_index_page(
        {
            "url": "https://nunatak.jobs.personio.de/job/2724247?language=de",
            "title": "(Senior) Machine Learning Engineer (m/w/d) | Jobs bei The Nunatak Group GmbH",
            "description": "",
        }
    )


def _personio_xml(positions) -> str:
    rows = []
    for pos in positions:
        extras = "".join(
            f"<office>{office}</office>" for office in pos.get("additionalOffices") or []
        )
        extra_xml = (
            f"<additionalOffices>{extras}</additionalOffices>" if extras else ""
        )
        descs = "".join(
            f"<jobDescription><name>x</name><value>{d}</value></jobDescription>"
            for d in pos.get("descriptions") or []
        )
        rows.append(
            "<position>"
            f"<id>{pos['id']}</id>"
            f"<subcompany>{pos.get('subcompany') or ''}</subcompany>"
            f"<office>{pos.get('office') or ''}</office>"
            f"{extra_xml}"
            f"<name>{pos.get('name') or ''}</name>"
            f"<jobDescriptions>{descs}</jobDescriptions>"
            f"<schedule>{pos.get('schedule') or ''}</schedule>"
            "</position>"
        )
    return '<?xml version="1.0" encoding="UTF-8"?><workzag-jobs>' + "".join(rows) + "</workzag-jobs>"


def test_listing_text_reads_personio_xml_remote_office(monkeypatch):
    engine = Engine()
    seen: list[str] = []
    xml = _personio_xml(
        [
            {
                "id": "2774857",
                "subcompany": "FINNOFLEET P-D-F GmbH",
                "office": "Berlin",
                "additionalOffices": ["Remote deutschlandweit"],
                "name": "AI Engineer (m/w/d) – Machine Learning &amp; GenAI",
                "descriptions": ["<p>Build AI.</p>"],
                "schedule": "full-time",
            }
        ]
    )

    async def fake_get(_client, url: str):
        seen.append(url)
        return xml

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    url = "https://finnofleet.jobs.personio.de/job/2774857?language=de"
    text = asyncio.run(engine._listing_text(url))
    assert seen == ["https://finnofleet.jobs.personio.de/xml"]
    from src.engine import _apply_listing, _html_is_index

    assert _html_is_index(text, url) is False
    opp = Opportunity(title="x", url=url)
    assert _apply_listing(opp, text) is False
    assert opp.company == "FINNOFLEET P-D-F GmbH"
    assert opp.remote is True
    assert opp.hours_per_week == 40
    assert "Build AI" in text


def test_personio_city_offices_stay_office():
    from src.engine import _apply_listing, _personio_to_html

    html = _personio_to_html(
        {
            "name": "(Senior) Machine Learning Engineer (m/w/d)",
            "subcompany": "The Nunatak Group GmbH",
            "offices": ["München", "Berlin"],
            "schedule": "full-time",
            "descriptions": ["<p>Consulting.</p>"],
        }
    )
    opp = Opportunity(
        title="x",
        url="https://nunatak.jobs.personio.de/job/2724247?language=de",
    )
    assert _apply_listing(opp, html) is False
    assert opp.company == "The Nunatak Group GmbH"
    assert opp.remote is False


def test_listing_text_personio_xml_missing_id_is_gone(monkeypatch):
    engine = Engine()
    seen: list[str] = []

    async def fake_get(_client, url: str):
        seen.append(url)
        return _personio_xml(
            [
                {
                    "id": "1111111",
                    "name": "Other role",
                    "office": "Berlin",
                    "subcompany": "Acme",
                }
            ]
        )

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    html = asyncio.run(
        engine._listing_text("https://acme.jobs.personio.de/job/2774857")
    )
    assert seen == ["https://acme.jobs.personio.de/xml"]
    assert html is None


def test_listing_text_personio_xml_404_falls_back_to_html(monkeypatch):
    engine = Engine()
    seen: list[str] = []

    async def fake_get(_client, url: str):
        seen.append(url)
        if url.endswith("/xml"):
            return None
        return (
            "<title>Senior AI Engineer (f/m/d) | Jobs at alpas</title>"
            '<script type="application/ld+json">'
            '{"@type":"JobPosting","title":"Senior AI Engineer (f/m/d)",'
            '"hiringOrganization":{"name":"alpas"}}'
            "</script><p>Berlin office.</p>"
        )

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    url = "https://alpas-gmbh.jobs.personio.de/job/2566758?language=en"
    text = asyncio.run(engine._listing_text(url))
    assert seen == [
        "https://alpas-gmbh.jobs.personio.de/xml",
        url,
    ]
    from src.engine import _apply_listing, _html_is_index

    assert _html_is_index(text, url) is False
    opp = Opportunity(title="x", url=url)
    _apply_listing(opp, text)
    assert opp.company == "alpas"


def test_html_is_index_keeps_personio_jobs_bei_role_title():
    from src.engine import _html_is_index

    html = (
        "<title>(Senior) Machine Learning Engineer (m/w/d) | "
        "Jobs bei The Nunatak Group GmbH</title><p>München</p>"
    )
    assert (
        _html_is_index(
            html, "https://nunatak.jobs.personio.de/job/2724247?language=de"
        )
        is False
    )
    assert _html_is_index(
        "<title>Jobs bei The Nunatak Group GmbH</title><p>Current openings</p>",
        "https://nunatak.jobs.personio.de/",
    )


def test_recruitee_job_urls_are_not_boards():
    from src.engine import _is_index_page, _lever_job_url, _recruitee_api_url

    apply_url = (
        "https://trafilea.recruitee.com/o/sr-software-engineer-fullstack/c/new"
    )
    assert _lever_job_url(apply_url) == (
        "https://trafilea.recruitee.com/o/sr-software-engineer-fullstack"
    )
    assert _recruitee_api_url(apply_url) == (
        "https://trafilea.recruitee.com/api/offers/sr-software-engineer-fullstack"
    )
    assert not _is_index_page(
        {
            "url": apply_url,
            "title": "Jobs at Trafilea",
            "description": "",
        }
    )
    assert _is_index_page(
        {
            "url": "https://holepunch.recruitee.com/",
            "title": "Holepunch - Career Portal",
            "description": "",
        }
    )
    assert _is_index_page(
        {
            "url": "https://skycellag.recruitee.com/homepage",
            "title": "Homepage [skycellag.recruitee.com]",
            "description": "",
        }
    )


def test_listing_text_reads_recruitee_api_not_form_pay_bands(monkeypatch):
    engine = Engine()
    seen: list[str] = []
    payload = {
        "offer": {
            "title": "React Native Developer - REMOTE",
            "company_name": "Gorin Systems",
            "slug": "reactjs-nodejs-developer-flutter-a-plus-remote-2",
            "remote": True,
            "hybrid": False,
            "on_site": False,
            "employment_type_code": "parttime_permanent",
            "salary": {"min": None, "max": None, "period": "hour", "currency": "USD"},
            "description": "<p>Build apps.</p>",
            "open_questions": [
                {
                    "body": "expected salary per annum",
                    "open_question_options": [
                        {"body": "$17,500 - $21,000"},
                        {"body": "$21,000 - $26,250"},
                    ],
                }
            ],
        }
    }

    async def fake_get(_client, url: str):
        seen.append(url)
        if "/api/offers/" in url:
            return json.dumps(payload)
        return (
            "<title>United States</title>"
            "<p>Choose closest * $17,500 - $21,000 $21,000 - $26,250</p>"
        )

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    url = (
        "https://gs.recruitee.com/o/"
        "reactjs-nodejs-developer-flutter-a-plus-remote-2"
    )
    text = asyncio.run(engine._listing_text(url))
    assert seen == [
        "https://gs.recruitee.com/api/offers/"
        "reactjs-nodejs-developer-flutter-a-plus-remote-2"
    ]
    from src.engine import _apply_listing

    opp = Opportunity(title="x", url=url)
    listed = _apply_listing(opp, text)
    assert listed is False
    assert opp.company == "Gorin Systems"
    assert opp.remote is True
    assert opp.pay_high is None
    assert "$17,500" not in text


def test_listing_text_recruitee_usd_salary_ranks(monkeypatch):
    engine = Engine()
    seen: list[str] = []
    payload = {
        "title": "Staff Engineer",
        "company_name": "Acme",
        "slug": "staff-engineer",
        "remote": True,
        "employment_type_code": "fulltime",
        "salary": {
            "min": 160000,
            "max": 190000,
            "period": "year",
            "currency": "USD",
        },
        "description": "<p>Distributed systems.</p>",
    }

    async def fake_get(_client, url: str):
        seen.append(url)
        return json.dumps(payload)

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    url = "https://acme.recruitee.com/o/staff-engineer"
    text = asyncio.run(engine._listing_text(url))
    assert seen == ["https://acme.recruitee.com/api/offers/staff-engineer"]
    from src.engine import _apply_listing

    opp = Opportunity(title="x", url=url)
    assert _apply_listing(opp, text) is True
    assert opp.company == "Acme"
    assert opp.pay_low == 160_000
    assert opp.pay_high == 190_000
    assert opp.hours_per_week == 40
    assert opp.score() == 95.0


def test_recruitee_zar_salary_is_foreign():
    from src.engine import _apply_listing, _foreign_salary, _recruitee_to_html

    html = _recruitee_to_html(
        {
            "title": "Senior .Net Developers Johannesburg",
            "company_name": "DVT",
            "remote": False,
            "on_site": True,
            "salary": {
                "min": "500",
                "max": "600",
                "period": "hour",
                "currency": "ZAR",
            },
            "description": "<p>Johannesburg office.</p>",
        }
    )
    opp = Opportunity(
        title="x", url="https://dvtcareers.recruitee.com/o/senior-net-developers-johannesburg"
    )
    _apply_listing(opp, html)
    assert opp.pay_high is None
    assert opp.remote is False
    assert _foreign_salary(html) is True


def test_listing_text_recruitee_api_404_is_gone(monkeypatch):
    engine = Engine()
    seen: list[str] = []

    async def fake_get(_client, url: str):
        seen.append(url)
        if "/api/offers/" in url:
            return None
        return "<title>United States</title><p>$17,500 - $21,000</p>"

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    html = asyncio.run(
        engine._listing_text(
            "https://nucsai.recruitee.com/o/senior-research-engineer-large-language-models"
        )
    )
    assert seen == [
        "https://nucsai.recruitee.com/api/offers/"
        "senior-research-engineer-large-language-models"
    ]
    assert html is None


def test_rippling_job_urls_are_not_boards():
    from src.engine import _is_index_page, _lever_job_url, _rippling_job_url

    apply = (
        "https://ats.rippling.com/button/jobs/"
        "d16f6e6a-1a1b-4ac6-a98d-2648b88e83f1/apply"
    )
    canonical = (
        "https://ats.rippling.com/button/jobs/"
        "d16f6e6a-1a1b-4ac6-a98d-2648b88e83f1"
    )
    assert _rippling_job_url(apply) == canonical
    assert _lever_job_url(apply) == canonical
    assert not _is_index_page(
        {"url": apply, "title": "Jobs at Button", "description": ""}
    )
    assert _is_index_page(
        {
            "url": "https://ats.rippling.com/button",
            "title": "Join us at Button",
            "description": "",
        }
    )
    assert _is_index_page(
        {
            "url": "https://www.rippling.com/careers",
            "title": "Rippling Careers | Work Will Never Be the Same",
            "description": "",
        }
    )


def _rippling_next_html(job_post=None, **api):
    data = {"jobBoard": {"companyName": "Button"}, **api}
    if job_post is not None:
        data["jobPost"] = job_post
    payload = {"props": {"pageProps": {"apiData": data}}}
    return f'<script id="__NEXT_DATA__">{json.dumps(payload)}</script>'


def test_listing_text_reads_rippling_next_data_pay(monkeypatch):
    engine = Engine()
    seen: list[str] = []
    html = _rippling_next_html(
        {
            "uuid": "d4d59ed8-4cc1-4fcf-ba37-b8273654bdf4",
            "name": "Senior Machine Learning Engineer – GeoAI Platform",
            "companyName": "Wherobots",
            "employmentType": {"label": "SALARIED_FT", "id": "Salaried, full-time"},
            "workLocations": ["Bellevue, WA", "San Francisco, CA"],
            "payRangeDetails": [
                {
                    "location": "SF, Seattle, Remote",
                    "currency": "USD",
                    "frequency": "YEAR",
                    "rangeStart": 185000,
                    "rangeEnd": 275000,
                    "isRemote": True,
                }
            ],
            "description": {"company": "<p>Geospatial.</p>", "role": "<p>Build GeoAI.</p>"},
        }
    )

    async def fake_get(_client, url: str):
        seen.append(url)
        return html

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    url = (
        "https://ats.rippling.com/wherobots/jobs/"
        "d4d59ed8-4cc1-4fcf-ba37-b8273654bdf4"
    )
    text = asyncio.run(engine._listing_text(url))
    assert seen == [url]
    from src.engine import _apply_listing

    opp = Opportunity(title="x", url=url)
    assert _apply_listing(opp, text) is True
    assert opp.company == "Wherobots"
    assert opp.pay_low == 185_000
    assert opp.pay_high == 275_000
    assert opp.hours_per_week == 40
    assert opp.remote is True
    assert opp.score() == 137.5


def test_listing_text_rippling_fills_company_without_inventing_pay(monkeypatch):
    engine = Engine()
    html = _rippling_next_html(
        {
            "uuid": "d16f6e6a-1a1b-4ac6-a98d-2648b88e83f1",
            "name": "Senior Machine Learning Engineer",
            "companyName": "Button",
            "employmentType": {"label": "SALARIED_FT", "id": "Salaried, full-time"},
            "workLocations": ["Remote (United States)"],
            "payRangeDetails": [],
            "description": {
                "role": "<p>The salary range for this role is expected to be between $153,000 - $198,000</p>"
            },
        }
    )

    async def fake_get(_client, url: str):
        return html

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    url = (
        "https://ats.rippling.com/button/jobs/"
        "d16f6e6a-1a1b-4ac6-a98d-2648b88e83f1"
    )
    text = asyncio.run(engine._listing_text(url))
    from src.engine import _apply_listing

    opp = Opportunity(title="x", url=url)
    assert _apply_listing(opp, text) is True
    assert opp.company == "Button"
    assert opp.remote is True
    assert opp.pay_low == 153_000
    assert opp.pay_high == 198_000


def test_rippling_office_pay_range_stays_office():
    from src.engine import _apply_listing, _rippling_to_html

    html = _rippling_to_html(
        {
            "name": "Engineer",
            "companyName": "Acme",
            "workLocations": ["Bellevue, WA"],
            "payRangeDetails": [
                {
                    "location": "Bellevue, WA",
                    "currency": "USD",
                    "frequency": "YEAR",
                    "rangeStart": 185000,
                    "rangeEnd": 275000,
                    "isRemote": False,
                }
            ],
        }
    )
    opp = Opportunity(
        title="x",
        url="https://ats.rippling.com/acme/jobs/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    )
    assert _apply_listing(opp, html) is True
    assert opp.remote is False
    assert opp.pay_high == 275_000


def test_rippling_cad_pay_range_is_foreign():
    from src.engine import _apply_listing, _foreign_salary, _rippling_to_html

    html = _rippling_to_html(
        {
            "name": "Engineer",
            "companyName": "Acme",
            "payRangeDetails": [
                {
                    "currency": "CAD",
                    "frequency": "YEAR",
                    "rangeStart": 160000,
                    "rangeEnd": 180000,
                }
            ],
        }
    )
    opp = Opportunity(
        title="x",
        url="https://ats.rippling.com/acme/jobs/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    )
    _apply_listing(opp, html)
    assert opp.pay_high is None
    assert _foreign_salary(html) is True


def test_listing_text_rippling_missing_job_is_gone(monkeypatch):
    engine = Engine()
    seen: list[str] = []

    async def fake_get(_client, url: str):
        seen.append(url)
        return _rippling_next_html()

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    html = asyncio.run(
        engine._listing_text(
            "https://ats.rippling.com/button/jobs/"
            "00000000-0000-0000-0000-000000000000"
        )
    )
    assert seen == [
        "https://ats.rippling.com/button/jobs/"
        "00000000-0000-0000-0000-000000000000"
    ]
    assert html is None


def test_breezy_job_urls_are_not_boards():
    from src.engine import _breezy_json_url, _is_index_page, _lever_job_url

    apply = (
        "https://concurrent-technologies-corporation.breezy.hr/p/"
        "4362758f2795-senior-software-engineer-specialist/apply"
    )
    assert _lever_job_url(apply) == (
        "https://concurrent-technologies-corporation.breezy.hr/p/4362758f2795"
    )
    assert _breezy_json_url(apply) == (
        "https://concurrent-technologies-corporation.breezy.hr/json"
    )
    assert not _is_index_page(
        {"url": apply, "title": "Jobs at Concurrent", "description": ""}
    )
    assert _is_index_page(
        {
            "url": "https://stacklet.breezy.hr/",
            "title": "Openings at Stacklet",
            "description": "",
        }
    )


def test_listing_text_reads_breezy_json_salary_not_board_html(monkeypatch):
    engine = Engine()
    seen: list[str] = []
    payload = [
        {
            "id": "a09e83b26526",
            "friendly_id": "a09e83b26526-senior-software-engineer-remote",
            "name": "Senior Software Engineer, remote",
            "type": {"id": "fullTime", "name": "Full-Time"},
            "location": {
                "is_remote": True,
                "name": "Austin, TX",
                "remote_details": {"value": "remote"},
            },
            "salary": "$135,000 – $195,000",
            "company": {"name": "Edfinity"},
        }
    ]

    async def fake_get(_client, url: str):
        seen.append(url)
        if url.endswith("/json"):
            return json.dumps(payload)
        return (
            "<title>%DOC_TITLE%Edfinity</title>"
            "<p>%POSITION_TYPE_FULL_TIME%$135,000 – $195,000%BUTTON</p>"
        )

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    url = "https://edfinity.breezy.hr/p/a09e83b26526-senior-software-engineer-remote"
    text = asyncio.run(engine._listing_text(url))
    assert seen == ["https://edfinity.breezy.hr/json"]
    from src.engine import _apply_listing

    opp = Opportunity(title="x", url=url)
    assert _apply_listing(opp, text) is True
    assert opp.company == "Edfinity"
    assert opp.remote is True
    assert opp.pay_low == 135_000
    assert opp.pay_high == 195_000
    assert opp.hours_per_week == 40
    assert opp.score() == 97.5
    assert "%DOC_TITLE%" not in text


def test_listing_text_breezy_missing_id_is_gone(monkeypatch):
    engine = Engine()
    seen: list[str] = []

    async def fake_get(_client, url: str):
        seen.append(url)
        if url.endswith("/json"):
            return json.dumps(
                [
                    {
                        "id": "a09e83b26526",
                        "name": "Senior Software Engineer, remote",
                        "salary": "$135,000 – $195,000",
                        "company": {"name": "Edfinity"},
                    }
                ]
            )
        return "<title>%DOC_TITLE%Edfinity</title><p>$135,000 – $195,000</p>"

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    html = asyncio.run(
        engine._listing_text(
            "https://edfinity.breezy.hr/p/00000000000000000000000000000000-gone-role"
        )
    )
    assert seen == ["https://edfinity.breezy.hr/json"]
    assert html is None


def test_breezy_foreign_salary_is_foreign():
    from src.engine import _apply_listing, _breezy_to_html, _foreign_salary

    html = _breezy_to_html(
        {
            "name": "Engineer",
            "company": {"name": "Acme"},
            "type": {"id": "fullTime"},
            "location": {"is_remote": True},
            "salary": "£60,000 – £80,000",
        }
    )
    opp = Opportunity(title="x", url="https://acme.breezy.hr/p/aaaaaaaaaaaa")
    _apply_listing(opp, html)
    assert opp.pay_high is None
    assert _foreign_salary(html) is True


def test_pinpoint_job_urls_are_not_boards():
    from src.engine import _is_index_page, _lever_job_url, _pinpoint_json_url

    localized = (
        "https://clearview.pinpointhq.com/en/postings/"
        "39dd4d7c-064e-400b-9857-06ae28cb6441"
    )
    canonical = (
        "https://clearview.pinpointhq.com/postings/"
        "39dd4d7c-064e-400b-9857-06ae28cb6441"
    )
    assert _lever_job_url(localized) == canonical
    assert _pinpoint_json_url(localized) == (
        "https://clearview.pinpointhq.com/postings.json"
    )
    assert not _is_index_page(
        {"url": localized, "title": "Jobs at Clearview AI", "description": ""}
    )
    assert _is_index_page(
        {
            "url": "https://rowdentech.pinpointhq.com/",
            "title": "Jobs at Rowden | Rowden Careers",
            "description": "",
        }
    )


def test_listing_text_reads_pinpoint_json_pay(monkeypatch):
    engine = Engine()
    seen: list[str] = []
    payload = {
        "data": [
            {
                "title": "Senior Machine Learning Engineer",
                "url": (
                    "https://clearview.pinpointhq.com/en/postings/"
                    "39dd4d7c-064e-400b-9857-06ae28cb6441"
                ),
                "path": "/en/postings/39dd4d7c-064e-400b-9857-06ae28cb6441",
                "employment_type": "full_time",
                "workplace_type": "hybrid",
                "compensation": "$180,000 - $250,000 / year",
                "compensation_minimum": 180000.0,
                "compensation_maximum": 250000.0,
                "compensation_currency": "USD",
                "compensation_frequency": "year",
                "location": {"name": "Remote USA"},
            }
        ]
    }

    async def fake_get(_client, url: str):
        seen.append(url)
        if url.endswith("/postings.json"):
            return json.dumps(payload)
        return "<title>Jobs at Clearview AI</title><p>Current openings</p>"

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    url = (
        "https://clearview.pinpointhq.com/postings/"
        "39dd4d7c-064e-400b-9857-06ae28cb6441"
    )
    text = asyncio.run(engine._listing_text(url))
    assert seen == ["https://clearview.pinpointhq.com/postings.json"]
    from src.engine import _apply_listing

    opp = Opportunity(title="x", url=url)
    assert _apply_listing(opp, text) is True
    assert opp.company == "Clearview"
    assert opp.remote is False
    assert opp.pay_low == 180_000
    assert opp.pay_high == 250_000
    assert opp.hours_per_week == 40
    assert opp.score() == 87.5


def test_listing_text_pinpoint_missing_id_is_gone(monkeypatch):
    engine = Engine()
    seen: list[str] = []

    async def fake_get(_client, url: str):
        seen.append(url)
        if url.endswith("/postings.json"):
            return json.dumps(
                {
                    "data": [
                        {
                            "title": "Other role",
                            "url": "https://emumba.pinpointhq.com/en/postings/"
                            "9d53ced8-835e-436b-8c63-6085d0495d4d",
                            "compensation": "$100,000 - $110,000 / year",
                            "compensation_minimum": 100000,
                            "compensation_maximum": 110000,
                            "compensation_currency": "USD",
                        }
                    ]
                }
            )
        return "<title>Jobs at Emumba</title><p>$100,000 - $110,000</p>"

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    html = asyncio.run(
        engine._listing_text(
            "https://emumba.pinpointhq.com/postings/"
            "edeb914d-2763-4d7c-99fa-46de8deb76f1"
        )
    )
    assert seen == ["https://emumba.pinpointhq.com/postings.json"]
    assert html is None


def test_pinpoint_gbp_pay_is_foreign():
    from src.engine import _apply_listing, _foreign_salary, _pinpoint_to_html

    html = _pinpoint_to_html(
        {
            "title": "Software Engineer",
            "workplace_type": "hybrid",
            "compensation": "£40,000 - £75,000 / year",
            "compensation_minimum": 40000,
            "compensation_maximum": 75000,
            "compensation_currency": "GBP",
            "compensation_frequency": "year",
        },
        "rowdentech",
    )
    opp = Opportunity(
        title="x",
        url="https://rowdentech.pinpointhq.com/postings/"
        "759d475f-0d95-4a1c-bda0-d8e3eba9f570",
    )
    _apply_listing(opp, html)
    assert opp.pay_high is None
    assert _foreign_salary(html) is True


def test_comeet_job_urls_are_not_boards():
    from src.engine import _comeet_job_url, _is_index_page

    job = (
        "https://www.comeet.com/jobs/vastdata/43.001/"
        "senior-software-engineer-platform/96.616"
    )
    assert _comeet_job_url(job) == job
    assert not _is_index_page(
        {"url": job, "title": "Jobs at VAST Data - Comeet", "description": ""}
    )
    assert not _is_index_page(
        {
            "url": (
                "https://www.comeet.com/jobs/aspectiva/35.000/"
                "senior-software-engineer/45.C50"
            ),
            "title": "Spark Hire Recruit Jobs | Spark Hire Recruit",
            "description": "",
        }
    )
    assert _is_index_page(
        {
            "url": "https://www.comeet.com/jobs/vastdata/43.001",
            "title": "Jobs at VAST Data - Comeet",
            "description": "",
        }
    )


def test_listing_text_reads_comeet_api_not_referral_pay(monkeypatch):
    engine = Engine()
    seen: list[str] = []
    page = '{"company_uid": "43.001", "token": "AABBCCDDEEFF00112233445566778899"}'
    payload = {
        "name": "Platform Senior Software Engineer",
        "company_name": "VAST Data",
        "employment_type": "Full-time",
        "workplace_type": "Hybrid",
        "location": {"name": "United States", "is_remote": True},
        "company_referrals_reward": "$2,000",
        "details": [
            {"name": "Description", "value": "<p>Build AI infrastructure.</p>"},
            {"name": "Referral reward", "value": "$2,000"},
        ],
    }

    async def fake_get(_client, url: str):
        seen.append(url)
        if "careers-api" in url:
            assert "AABBCCDDEEFF00112233445566778899" in url
            return json.dumps(payload)
        return page

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    url = (
        "https://www.comeet.com/jobs/vastdata/43.001/"
        "senior-software-engineer-platform/96.616"
    )
    text = asyncio.run(engine._listing_text(url))
    assert seen[0] == url
    assert "careers-api/2.0/company/43.001/positions/96.616" in seen[1]
    from src.engine import _apply_listing

    opp = Opportunity(title="x", url=url)
    listed = _apply_listing(opp, text)
    assert listed is False
    assert opp.company == "VAST Data"
    assert opp.remote is True
    assert opp.pay_high is None
    assert opp.hours_per_week == 40
    assert "$2,000" not in text


def test_listing_text_comeet_missing_position_is_gone(monkeypatch):
    engine = Engine()
    seen: list[str] = []

    async def fake_get(_client, url: str):
        seen.append(url)
        if "careers-api" in url:
            return None
        return '{"token": "AABBCCDDEEFF00112233445566778899"}'

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    html = asyncio.run(
        engine._listing_text(
            "https://www.comeet.com/jobs/mentee_robotics/6A.002/"
            "senior-software-engineer-ai-infra/01.852"
        )
    )
    assert any("careers-api" in u for u in seen)
    assert html is None


def test_bamboohr_job_urls_are_not_boards():
    from src.engine import _bamboohr_detail_url, _is_index_page, _lever_job_url

    job = "https://selectorsoftware.bamboohr.com/careers/157/detail"
    assert _lever_job_url(job) == "https://selectorsoftware.bamboohr.com/careers/157"
    assert _bamboohr_detail_url(job) == (
        "https://selectorsoftware.bamboohr.com/careers/157/detail"
    )
    assert not _is_index_page(
        {"url": job, "title": "selectorsoftware.bamboohr.com", "description": ""}
    )
    assert _is_index_page(
        {
            "url": "https://sixworks.bamboohr.com/careers/list",
            "title": "sixworks.bamboohr.com",
            "description": "",
        }
    )
    assert _is_index_page(
        {
            "url": "https://www.bamboohr.com/careers/engineering-it-team",
            "title": "Engineering & IT Careers | BambooHR",
            "description": "",
        }
    )


def test_listing_text_reads_bamboohr_detail_not_form_pay(monkeypatch):
    engine = Engine()
    seen: list[str] = []
    payload = {
        "result": {
            "jobOpening": {
                "jobOpeningName": "Product Manager – AIOps",
                "employmentStatusLabel": "Full-Time",
                "locationType": 0,
                "location": {
                    "city": "Santa Clara",
                    "state": "California",
                    "addressCountry": "United States",
                },
                "compensation": "$180,000 - $220,000",
                "description": "<p>Build AIOps products.</p>",
            },
            "formFields": {
                "desiredPay": {
                    "isRequired": False,
                    "value": "$17,500",
                    "label": "Desired Pay",
                }
            },
        }
    }

    async def fake_get(_client, url: str):
        seen.append(url)
        return json.dumps(payload)

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    url = "https://selectorsoftware.bamboohr.com/careers/157"
    text = asyncio.run(engine._listing_text(url))
    assert seen == ["https://selectorsoftware.bamboohr.com/careers/157/detail"]
    from src.engine import _apply_listing

    opp = Opportunity(title="x", url=url)
    listed = _apply_listing(opp, text)
    assert listed is True
    assert opp.company == "Selectorsoftware"
    assert opp.remote is False
    assert opp.pay_low == 180_000
    assert opp.pay_high == 220_000
    assert opp.hours_per_week == 40
    assert "$17,500" not in text


def test_listing_text_bamboohr_hybrid_is_office_and_remote_type(monkeypatch):
    engine = Engine()

    async def fake_get(_client, url: str):
        if url.endswith("/112/detail"):
            return json.dumps(
                {
                    "result": {
                        "jobOpening": {
                            "jobOpeningName": "Customer Success Manager",
                            "employmentStatusLabel": "Full-Time",
                            "locationType": 2,
                            "location": {"city": "Santa Clara", "state": "California"},
                            "compensation": "$130K to $170K",
                            "description": "<p>Hybrid CSM role.</p>",
                        }
                    }
                }
            )
        return json.dumps(
            {
                "result": {
                    "jobOpening": {
                        "jobOpeningName": "Channel Account Manager",
                        "employmentStatusLabel": "Full-Time",
                        "locationType": 1,
                        "atsLocation": {
                            "country": "United States",
                            "state": "New York",
                            "city": "New York",
                        },
                        "compensation": "$225,000 - $260,000",
                        "description": "<p>Remote, field-based.</p>",
                    }
                }
            }
        )

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    from src.engine import _apply_listing

    hybrid = Opportunity(
        title="x", url="https://selectorsoftware.bamboohr.com/careers/112"
    )
    _apply_listing(
        hybrid,
        asyncio.run(engine._listing_text(hybrid.url)),
    )
    assert hybrid.remote is False
    assert hybrid.pay_low == 130_000
    assert hybrid.pay_high == 170_000

    remote = Opportunity(title="x", url="https://alkira.bamboohr.com/careers/234")
    _apply_listing(remote, asyncio.run(engine._listing_text(remote.url)))
    assert remote.remote is True
    assert remote.pay_low == 225_000
    assert remote.pay_high == 260_000
    assert remote.company == "Alkira"


def test_listing_text_bamboohr_missing_id_is_gone(monkeypatch):
    engine = Engine()
    seen: list[str] = []

    async def fake_get(_client, url: str):
        seen.append(url)
        return None

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    html = asyncio.run(
        engine._listing_text("https://lawsonlundell.bamboohr.com/careers/244")
    )
    assert seen == ["https://lawsonlundell.bamboohr.com/careers/244/detail"]
    assert html is None


def test_jazzhr_job_urls_are_not_boards():
    from src.engine import _is_index_page, _jazzhr_job_url, _lever_job_url

    job = (
        "https://harvesthosts.applytojob.com/apply/"
        "etlianxvgi/senior-full-stack-engineer"
    )
    assert _lever_job_url(job) == (
        "https://harvesthosts.applytojob.com/apply/etlianxvgi"
    )
    assert _jazzhr_job_url(job) == (
        "https://harvesthosts.applytojob.com/apply/etlianxvgi"
    )
    assert not _is_index_page(
        {"url": job, "title": "Senior Full Stack Engineer - Harvest Hosts - Career Page", "description": ""}
    )
    assert _is_index_page(
        {
            "url": "https://veronetworks.applytojob.com/apply/",
            "title": "Vero Networks - Career Page",
            "description": "",
        }
    )
    assert _is_index_page(
        {
            "url": "https://usengineering.applytojob.com/apply",
            "title": "U.S. Engineering - Career Page",
            "description": "",
        }
    )


def test_listing_text_reads_jazzhr_jobposting_pay(monkeypatch):
    engine = Engine()
    seen: list[str] = []
    html = (
        "<title>Senior Full Stack Engineer - Harvest Hosts - Career Page</title>"
        '<script type="application/ld+json">'
        '{"@type":"JobPosting","title":"Senior Full Stack Engineer",'
        '"hiringOrganization":{"name":"Harvest Hosts"},'
        '"employmentType":"FULL_TIME","jobLocationType":"TELECOMMUTE",'
        '"baseSalary":{"currency":"USD","value":{"unitText":"YEAR",'
        '"minValue":130000,"maxValue":150000}}}'
        "</script>"
        "<p>Remote</p>"
    )

    async def fake_get(_client, url: str):
        seen.append(url)
        return html

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    url = "https://harvesthosts.applytojob.com/apply/etlianxvgi/senior-full-stack-engineer"
    text = asyncio.run(engine._listing_text(url))
    assert seen == ["https://harvesthosts.applytojob.com/apply/etlianxvgi"]
    from src.engine import _apply_listing

    opp = Opportunity(title="x", url=url)
    listed = _apply_listing(opp, text)
    assert listed is True
    assert opp.company == "Harvest Hosts"
    assert opp.remote is True
    assert opp.pay_low == 130_000
    assert opp.pay_high == 150_000
    assert opp.hours_per_week == 40


def test_listing_text_jazzhr_board_shell_is_gone(monkeypatch):
    engine = Engine()
    seen: list[str] = []
    html = (
        "<title>Anika Systems - Career Page</title>"
        '<script type="application/ld+json">'
        '{"@type":"Organization","name":"Anika Systems"}'
        "</script>"
        "<p>Full Time</p>"
    )

    async def fake_get(_client, url: str):
        seen.append(url)
        return html

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    gone = asyncio.run(
        engine._listing_text(
            "https://anikasystems.applytojob.com/apply/vQZOpYzJri/AI-Engineer"
        )
    )
    assert seen == ["https://anikasystems.applytojob.com/apply/vQZOpYzJri"]
    assert gone is None


def test_listing_text_jazzhr_410_is_gone(monkeypatch):
    engine = Engine()
    seen: list[str] = []

    async def fake_get(_client, url: str):
        seen.append(url)
        return None

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    html = asyncio.run(
        engine._listing_text(
            "https://veronetworks.applytojob.com/apply/zzzzzzzzzz/Gone-Role"
        )
    )
    assert seen == ["https://veronetworks.applytojob.com/apply/zzzzzzzzzz"]
    assert html is None


def test_dover_job_urls_are_not_boards():
    from src.engine import _dover_api_url, _is_index_page, _lever_job_url

    job = (
        "https://app.dover.com/apply/conveyor/"
        "1ba85ce7-7f2e-4230-ba94-4bb67a3371b8"
    )
    assert _lever_job_url(job) == job
    assert _dover_api_url(job) == (
        "https://app.dover.com/api/v1/inbound/application-portal-job/"
        "1ba85ce7-7f2e-4230-ba94-4bb67a3371b8"
    )
    assert not _is_index_page(
        {"url": job, "title": "Dover", "description": ""}
    )
    assert _is_index_page(
        {
            "url": "https://app.dover.com/jobs/causallabs",
            "title": "Careers - app.dover.com",
            "description": "",
        }
    )
    assert _is_index_page(
        {
            "url": "https://app.dover.com/apply/SemiAnalysis",
            "title": "Dover",
            "description": "",
        }
    )


def test_listing_text_reads_dover_api_pay_not_form_questions(monkeypatch):
    engine = Engine()
    seen: list[str] = []
    payload = {
        "id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        "active": True,
        "is_private": False,
        "client_name": "Acme",
        "title": "Staff AI Engineer",
        "workplace_type": "REMOTE",
        "locations": [{"location_type": "REMOTE", "name": "United States"}],
        "compensation": {
            "lower_bound": 180000,
            "upper_bound": 220000,
            "currency_code": "USD",
            "salary_range_type": "YEARLY",
            "employment_type": "FULL_TIME",
        },
        "application_questions": [
            {"name": "desired_salary", "label": "Expected pay", "value": "$17,500"}
        ],
        "user_provided_description": "<p>Build AI systems.</p>",
    }

    async def fake_get(_client, url: str):
        seen.append(url)
        return json.dumps(payload)

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    url = "https://app.dover.com/apply/Acme/aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    text = asyncio.run(engine._listing_text(url))
    assert seen == [
        "https://app.dover.com/api/v1/inbound/application-portal-job/"
        "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    ]
    from src.engine import _apply_listing

    opp = Opportunity(title="Dover", url=url)
    listed = _apply_listing(opp, text)
    assert listed is True
    assert opp.company == "Acme"
    assert opp.remote is True
    assert opp.pay_low == 180_000
    assert opp.pay_high == 220_000
    assert opp.hours_per_week == 40
    assert "$17,500" not in text


def test_dover_inr_salary_is_foreign():
    from src.engine import _apply_listing, _dover_to_html, _foreign_salary

    html = _dover_to_html(
        {
            "title": "ML Engineer Intern",
            "client_name": "Peakflo",
            "workplace_type": "REMOTE",
            "compensation": {
                "lower_bound": 480000,
                "upper_bound": 600000,
                "currency_code": "INR",
                "salary_range_type": "YEARLY",
                "employment_type": "INTERNSHIP",
            },
            "user_provided_description": "<p>India remote intern.</p>",
        }
    )
    opp = Opportunity(
        title="x",
        url="https://app.dover.com/apply/Peakflo/f7345aa2-9bc2-4196-99e4-2c3f277f9bfb",
    )
    listed = _apply_listing(opp, html)
    assert listed is False
    assert opp.pay_high is None
    assert opp.remote is True
    assert _foreign_salary(html) is True


def test_listing_text_dover_missing_id_is_gone(monkeypatch):
    engine = Engine()
    seen: list[str] = []

    async def fake_get(_client, url: str):
        seen.append(url)
        return None

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    html = asyncio.run(
        engine._listing_text(
            "https://app.dover.com/dover/careers/"
            "81955b55-70c1-435d-860e-050487e2eae2"
        )
    )
    assert seen == [
        "https://app.dover.com/api/v1/inbound/application-portal-job/"
        "81955b55-70c1-435d-860e-050487e2eae2"
    ]
    assert html is None


def test_listing_text_dover_inactive_is_gone(monkeypatch):
    engine = Engine()

    async def fake_get(_client, url: str):
        return json.dumps(
            {
                "id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                "active": False,
                "title": "Closed Role",
                "client_name": "Acme",
            }
        )

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    html = asyncio.run(
        engine._listing_text(
            "https://app.dover.com/apply/Acme/bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
        )
    )
    assert html is None


def test_gem_job_urls_are_not_boards():
    from src.engine import _gem_job_url, _is_index_page, _lever_job_url

    job = (
        "https://jobs.gem.com/ascendarc/"
        "am9icG9zdDr7i9rbOpLD20JgJavBiLRk/application"
    )
    assert _lever_job_url(job) == (
        "https://jobs.gem.com/ascendarc/am9icG9zdDr7i9rbOpLD20JgJavBiLRk"
    )
    assert _gem_job_url(job) == (
        "https://jobs.gem.com/ascendarc/am9icG9zdDr7i9rbOpLD20JgJavBiLRk"
    )
    assert not _is_index_page(
        {"url": job, "title": "Senior RF Board Engineer", "description": ""}
    )
    assert _is_index_page(
        {
            "url": "https://jobs.gem.com/ascendarc",
            "title": "AscendArc Careers",
            "description": "",
        }
    )


def test_listing_text_reads_gem_graphql_pay(monkeypatch):
    engine = Engine()
    seen: list[tuple[str, str]] = []
    payload = {
        "title": "Senior RF Board Engineer",
        "descriptionHtml": "<p>The salary range for this position is $125,000-210,000.</p>",
        "compensationHtml": None,
        "isUnlistedExternally": False,
        "locations": [
            {"name": "Beaverton", "city": "Beaverton", "isoCountry": "USA", "isRemote": False}
        ],
        "job": {
            "locationType": "IN_OFFICE",
            "employmentType": "FULL_TIME",
            "teamDisplayName": "AscendArc",
        },
    }

    async def fake_gem(_client, board: str, jid: str):
        seen.append((board, jid))
        return payload

    async def fake_get(_client, _url: str):
        raise AssertionError("SPA HTML must not be fetched when GraphQL returns a posting")

    monkeypatch.setattr("src.engine._gem_posting", fake_gem)
    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    url = (
        "https://jobs.gem.com/ascendarc/am9icG9zdDr7i9rbOpLD20JgJavBiLRk"
    )
    text = asyncio.run(engine._listing_text(url))
    assert seen == [("ascendarc", "am9icG9zdDr7i9rbOpLD20JgJavBiLRk")]
    from src.engine import _apply_listing

    opp = Opportunity(title="x", url=url)
    listed = _apply_listing(opp, text)
    assert listed is True
    assert opp.company == "AscendArc"
    assert opp.remote is False
    assert opp.pay_low == 125_000
    assert opp.pay_high == 210_000
    assert opp.hours_per_week == 40


def test_listing_text_gem_graphql_null_is_gone(monkeypatch):
    engine = Engine()
    seen: list[tuple[str, str]] = []

    async def fake_gem(_client, board: str, jid: str):
        seen.append((board, jid))
        return None

    async def fake_get(_client, _url: str):
        raise AssertionError("SPA HTML must not be fetched when GraphQL says gone")

    monkeypatch.setattr("src.engine._gem_posting", fake_gem)
    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    html = asyncio.run(
        engine._listing_text("https://jobs.gem.com/aux-insights/4141242008")
    )
    assert seen == [("aux-insights", "4141242008")]
    assert html is None


def test_listing_text_gem_graphql_timeout_is_empty(monkeypatch):
    engine = Engine()

    async def fake_gem(_client, _board: str, _jid: str):
        return {}

    async def fake_get(_client, _url: str):
        raise AssertionError("SPA HTML must not be fetched when GraphQL times out")

    monkeypatch.setattr("src.engine._gem_posting", fake_gem)
    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    html = asyncio.run(
        engine._listing_text(
            "https://jobs.gem.com/converge/am9icG9zdDqFHmjnR-_vQgjRElAPit0P"
        )
    )
    assert html == ""


def test_walmart_ids_and_board():
    from src.engine import (
        _ats_job_url,
        _company_from_url,
        _is_index_page,
        _walmart_ids,
        _walmart_is_board,
        _walmart_job_url,
    )

    url = "https://careers.walmart.com/us/en/jobs/R-2395925"
    assert _walmart_ids(url) == "R-2395925"
    assert _walmart_job_url(url + "?foo=1") == url
    assert _ats_job_url(url)
    assert _company_from_url(url) == "Walmart"
    assert not _walmart_is_board(url)
    assert _walmart_is_board("https://careers.walmart.com/us/en")
    assert _is_index_page(
        {"url": "https://careers.walmart.com/us/en", "title": "Careers", "description": ""}
    )
    assert not _is_index_page(
        {"url": url, "title": "Jobs at Walmart", "description": ""}
    )


def _walmart_next_html(details: dict, job_id: str = "R-1") -> str:
    payload = {"props": {"pageProps": {"jobId": job_id, "jobDetails": details}}}
    return f'<script id=__NEXT_DATA__ type="application/json">{json.dumps(payload)}</script>'


def test_walmart_details_inactive_is_gone():
    from src.engine import _walmart_details

    html = _walmart_next_html(
        {
            "title": "Staff, Machine Learning Engineer",
            "brand": "Walmart",
            "active": False,
            "positionAvailable": 0,
            "payRange": [{"location": "Bentonville, Arkansas", "min": "130000.00", "max": "260000.00"}],
        },
        "R-2395925",
    )
    assert _walmart_details(html, "R-2395925") is None
    open_html = _walmart_next_html(
        {
            "title": "Staff ML",
            "brand": "Walmart",
            "active": True,
            "positionAvailable": 1,
            "primaryLocation": {"city": "BENTONVILLE", "stateCode": "AR"},
            "payRange": [{"location": "Bentonville, Arkansas", "min": "130000.00", "max": "260000.00"}],
            "payPlanData": {"currencyReference": {"currencyId": "USD"}},
        },
        "R-1",
    )
    job = _walmart_details(open_html, "R-1")
    assert job and job["brand"] == "Walmart"


def test_walmart_to_html_fills_company_office_pay():
    from src.engine import _apply_listing, _foreign_salary, _walmart_to_html

    html = _walmart_to_html(
        {
            "title": "Staff, Machine Learning Engineer",
            "jobPostingTitle": "Staff, Machine Learning Engineer",
            "brand": "Walmart",
            "primaryLocation": {"city": "BENTONVILLE", "stateCode": "AR"},
            "additionalLocations": [{"city": "Sunnyvale", "stateCode": "CA"}],
            "payRange": [
                {"location": "Bentonville, Arkansas", "min": "130000.00", "max": "260000.00"},
                {"location": "Sunnyvale, California", "min": "169000.00", "max": "338000.00"},
            ],
            "payPlanData": {"currencyReference": {"currencyId": "USD"}},
            "description": "<p>Build models onsite.</p>",
        }
    )
    opp = Opportunity(
        title="x",
        url="https://careers.walmart.com/us/en/jobs/R-1",
    )
    _apply_listing(opp, html)
    assert opp.company == "Walmart"
    assert opp.title == "Staff, Machine Learning Engineer"
    assert opp.remote is False
    assert opp.pay_low == 130_000
    assert opp.pay_high == 260_000
    assert opp.score() == 91.0
    assert _foreign_salary(html) is False


def test_walmart_to_html_foreign_currency_is_not_usd():
    from src.engine import _apply_listing, _foreign_salary, _walmart_to_html

    html = _walmart_to_html(
        {
            "title": "Engineer",
            "brand": "Walmart",
            "primaryLocation": {"city": "Mexico City", "stateCode": ""},
            "payRange": [{"min": "500000.00", "max": "700000.00"}],
            "payPlanData": {"currencyReference": {"currencyId": "MXN"}},
        }
    )
    opp = Opportunity(title="x", url="https://careers.walmart.com/us/en/jobs/R-2")
    _apply_listing(opp, html)
    assert opp.pay_high is None
    assert _foreign_salary(html) is True


def test_listing_text_walmart_inactive_is_gone(monkeypatch):
    engine = Engine()

    async def fake_get(_client, url: str):
        assert "careers.walmart.com" in url
        return _walmart_next_html(
            {
                "title": "Staff, Machine Learning Engineer",
                "brand": "Walmart",
                "active": False,
                "positionAvailable": 0,
                "payRange": [{"min": "130000.00", "max": "260000.00"}],
            },
            "R-2395925",
        )

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    html = asyncio.run(
        engine._listing_text("https://careers.walmart.com/us/en/jobs/R-2395925")
    )
    assert html is None


def test_listing_text_walmart_open_reads_next_data(monkeypatch):
    engine = Engine()

    async def fake_get(_client, url: str):
        return _walmart_next_html(
            {
                "title": "Staff ML",
                "brand": "Walmart",
                "active": True,
                "positionAvailable": 2,
                "primaryLocation": {"city": "BENTONVILLE", "stateCode": "AR"},
                "payRange": [{"location": "Bentonville, Arkansas", "min": "130000.00", "max": "260000.00"}],
                "payPlanData": {"currencyReference": {"currencyId": "USD"}},
            },
            "R-9",
        )

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    url = "https://careers.walmart.com/us/en/jobs/R-9"
    html = asyncio.run(engine._listing_text(url))
    from src.engine import _apply_listing

    opp = Opportunity(title="x", url=url)
    _apply_listing(opp, html)
    assert opp.company == "Walmart"
    assert opp.pay_high == 260_000
    assert opp.remote is False


def _apple_hydration_html(payload: dict) -> str:
    blob = json.dumps(payload)
    escaped = json.dumps(blob)[1:-1]
    return (
        f'<script>window.__staticRouterHydrationData = JSON.parse("{escaped}");</script>'
    )


def test_apple_ids_keep_jobs_titled_postings():
    from src.engine import (
        _apple_ids,
        _apple_is_board,
        _apple_job_url,
        _ats_job_url,
        _company_from_url,
        _is_index_page,
    )

    url = (
        "https://jobs.apple.com/en-us/details/200617298-1435/"
        "aiml-staff-machine-learning-engineer"
    )
    assert _apple_ids(url) == "200617298-1435"
    assert _apple_job_url(url) == "https://jobs.apple.com/en-us/details/200617298-1435"
    assert _ats_job_url(url)
    assert _company_from_url(url) == "Apple"
    assert not _is_index_page(
        {
            "url": url,
            "title": "AIML - Staff Machine Learning Engineer - Jobs - Careers at Apple",
            "description": "",
        }
    )
    assert _apple_is_board("https://jobs.apple.com/en-us/search?search=ml")
    assert _is_index_page(
        {
            "url": "https://jobs.apple.com/en-us/search?search=ml",
            "title": "Search Jobs - Jobs - Careers at Apple",
            "description": "",
        }
    )


def test_apple_job_hydration_404_is_gone():
    from src.engine import _apple_job

    html = _apple_hydration_html(
        {
            "loaderData": {"root": {}},
            "errors": {
                "jobDetails": {
                    "status": 404,
                    "statusText": "",
                    "internal": False,
                    "data": "",
                    "__type": "RouteErrorResponse",
                }
            },
        }
    )
    assert _apple_job(html) is None
    assert _apple_job("Page not found. Sorry, this role does not exist or is no longer available.") is None


def test_apple_to_html_fills_company_hours_and_office():
    from src.engine import _apply_listing, _apple_to_html

    html = _apple_to_html(
        {
            "postingTitle": "Packaging Product Design Engineer",
            "homeOffice": False,
            "standardWeeklyHours": 40,
            "locations": [{"name": "Austin", "city": "Austin", "countryName": "United States"}],
            "jobSummary": "Design packaging in Austin.",
        }
    )
    opp = Opportunity(title="x", url="https://jobs.apple.com/en-us/details/200681889-0157")
    _apply_listing(opp, html)
    assert opp.company == "Apple"
    assert opp.title == "Packaging Product Design Engineer"
    assert opp.remote is False
    assert opp.hours_per_week == 40
    assert opp.pay_high is None
    remote_html = _apple_to_html(
        {
            "postingTitle": "Engineer",
            "homeOffice": True,
            "jobSummary": "Work from home.",
        }
    )
    remote = Opportunity(title="x", url="https://jobs.apple.com/en-us/details/1")
    _apply_listing(remote, remote_html)
    assert remote.remote is True
    assert remote.company == "Apple"


def test_listing_text_apple_gone_is_none(monkeypatch):
    engine = Engine()

    async def fake_get(_client, url: str):
        assert "jobs.apple.com" in url
        return _apple_hydration_html(
            {
                "loaderData": {"root": {}},
                "errors": {"jobDetails": {"status": 404, "internal": False, "data": ""}},
            }
        )

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    html = asyncio.run(
        engine._listing_text(
            "https://jobs.apple.com/en-us/details/200617298-1435/aiml-staff-machine-learning-engineer"
        )
    )
    assert html is None


def test_listing_text_apple_open_reads_jobs_data(monkeypatch):
    engine = Engine()

    async def fake_get(_client, url: str):
        return _apple_hydration_html(
            {
                "loaderData": {
                    "root": {},
                    "jobDetails": {
                        "jobsData": {
                            "id": "200681889-0157",
                            "postingTitle": "Packaging Product Design Engineer",
                            "homeOffice": False,
                            "standardWeeklyHours": 40,
                            "locations": [{"name": "Austin", "city": "Austin"}],
                            "jobSummary": "Design packaging.",
                        }
                    },
                }
            }
        )

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    url = "https://jobs.apple.com/en-us/details/200681889-0157"
    html = asyncio.run(engine._listing_text(url))
    from src.engine import _apply_listing

    opp = Opportunity(title="x", url=url)
    _apply_listing(opp, html)
    assert opp.company == "Apple"
    assert opp.title == "Packaging Product Design Engineer"
    assert opp.hours_per_week == 40
    assert opp.remote is False


def test_listing_plain_text_drops_related_job_pay_and_foreign_cards():
    from src.engine import _apply_listing, _foreign_salary, _listing_plain_text, _parse_pay

    html = """
    <title>IT Security Administrator at Bitwarden</title>
    <p>$115,000 - $145,000 a year. This is an all-remote team.</p>
    <h2>Similar Jobs</h2>
    <p>DevOps Engineer ₹6L – ₹9L</p>
    <p>Renewals Manager $422,000 - $502,000</p>
    """
    other = """
    <title>Engineer</title>
    <p>Great team. Apply now.</p>
    <h2>Other Jobs</h2>
    <p>Account Executive $180,000 - $220,000</p>
    """
    featured = """
    <title>Engineer</title>
    <p>Great team. Apply now.</p>
    <h2>Featured Jobs</h2>
    <p>Account Executive $180,000 - $220,000</p>
    """
    viewed = """
    <title>Engineer</title>
    <p>Great team. Apply now.</p>
    <h2>People also viewed</h2>
    <p>Account Executive $180,000 - $220,000</p>
    """
    positions = """
    <title>Engineer</title>
    <p>Great team. Apply now.</p>
    <h2>Other positions</h2>
    <p>Account Executive $180,000 - $220,000</p>
    """
    opportunities = """
    <title>Engineer</title>
    <p>Great team. Apply now.</p>
    <h2>Other opportunities</h2>
    <p>Account Executive $180,000 - $220,000</p>
    """
    might = """
    <title>Engineer</title>
    <p>Great team. Apply now.</p>
    <h2>You might also like</h2>
    <p>Account Executive $180,000 - $220,000</p>
    """
    applicants = """
    <title>Engineer</title>
    <p>Great team. Apply now.</p>
    <h2>Applicants also viewed</h2>
    <p>Account Executive $180,000 - $220,000</p>
    """
    featured_roles = """
    <title>Engineer</title>
    <p>Great team. Apply now.</p>
    <h2>Featured roles</h2>
    <p>Account Executive $180,000 - $220,000</p>
    """
    own_then_opps = """
    <title>Engineer</title>
    <p>$115,000 - $145,000 a year.</p>
    <h2>Other opportunities</h2>
    <p>Account Executive $180,000 - $220,000</p>
    """
    glued = (
        "<title>IT Security Administrator at Bitwarden</title>"
        "<p>$115,000 - $145,000 a year.</p>"
        "<p>historySimilar JobsDrona Pay DevOps Engineer ₹6L – ₹9L Renewals Manager $422,000</p>"
    )
    glued_other = (
        "<title>Engineer</title>"
        "<p>Great team. Apply now.</p>"
        "<p>historyOther Jobs Account Executive $180,000 - $220,000</p>"
    )
    assert _parse_pay(_listing_plain_text(html)) == (115_000, 145_000)
    assert _parse_pay(_listing_plain_text(glued)) == (115_000, 145_000)
    assert _parse_pay(_listing_plain_text(other)) == (None, None)
    assert _parse_pay(_listing_plain_text(featured)) == (None, None)
    assert _parse_pay(_listing_plain_text(viewed)) == (None, None)
    assert _parse_pay(_listing_plain_text(glued_other)) == (None, None)
    assert _parse_pay(_listing_plain_text(positions)) == (None, None)
    assert _parse_pay(_listing_plain_text(opportunities)) == (None, None)
    assert _parse_pay(_listing_plain_text(might)) == (None, None)
    assert _parse_pay(_listing_plain_text(applicants)) == (None, None)
    assert _parse_pay(_listing_plain_text(featured_roles)) == (None, None)
    assert _parse_pay(_listing_plain_text(own_then_opps)) == (115_000, 145_000)
    assert _foreign_salary(html) is False
    assert _foreign_salary(glued) is False
    opp = Opportunity(
        title="IT Security Administrator",
        url="https://wellfound.com/jobs/4335648-it-security-administrator",
    )
    _apply_listing(opp, html)
    assert opp.pay_low == 115_000
    assert opp.pay_high == 145_000
    assert opp.company == "Bitwarden"
    ghost = Opportunity(title="Engineer", url="https://jobs.example/other")
    assert _apply_listing(ghost, other) is False
    assert ghost.pay_high is None
    feat = Opportunity(title="Engineer", url="https://jobs.example/featured")
    assert _apply_listing(feat, featured) is False
    assert feat.pay_high is None
    pos = Opportunity(title="Engineer", url="https://jobs.example/positions")
    assert _apply_listing(pos, positions) is False
    assert pos.pay_high is None
    opps = Opportunity(title="Engineer", url="https://jobs.example/opps")
    assert _apply_listing(opps, opportunities) is False
    assert opps.pay_high is None
    kept = Opportunity(title="Engineer", url="https://jobs.example/kept")
    assert _apply_listing(kept, own_then_opps) is True
    assert kept.pay_high == 145_000
    growth = Opportunity(title="Engineer", url="https://jobs.example/growth")
    assert _apply_listing(
        growth,
        "<title>Engineer</title><p>We offer more opportunities for growth. Salary $180,000</p>",
    ) is True
    assert growth.pay_high == 180_000
    browse = Opportunity(title="Engineer", url="https://jobs.example/browse")
    assert _apply_listing(
        browse,
        "<title>Engineer</title><p>Great team. Apply now.</p>"
        "<h2>Browse jobs</h2><p>Account Executive $180,000 - $220,000</p>",
    ) is False
    assert browse.pay_high is None
    latest = Opportunity(title="Engineer", url="https://jobs.example/latest")
    assert _apply_listing(
        latest,
        "<title>Engineer</title><p>Great team. Apply now.</p>"
        "<h2>Latest jobs</h2><p>Account Executive $180,000 - $220,000</p>",
    ) is False
    assert latest.pay_high is None
    see = Opportunity(title="Engineer", url="https://jobs.example/see")
    assert _apply_listing(
        see,
        "<title>Engineer</title><p>Great team. Apply now.</p>"
        "<h2>See also</h2><p>Account Executive $180,000 - $220,000</p>",
    ) is False
    assert see.pay_high is None
    more = Opportunity(title="Engineer", url="https://jobs.example/more-opps")
    assert _apply_listing(
        more,
        "<title>Engineer</title><p>Great team. Apply now.</p>"
        "<h2>More opportunities</h2><p>Account Executive $180,000 - $220,000</p>",
    ) is False
    assert more.pay_high is None
    own_browse = Opportunity(title="Engineer", url="https://jobs.example/own-browse")
    assert _apply_listing(
        own_browse,
        "<title>Engineer</title><p>$115,000 - $145,000 a year.</p>"
        "<h2>Browse jobs</h2><p>Account Executive $180,000 - $220,000</p>",
    ) is True
    assert own_browse.pay_high == 145_000
    nav = Opportunity(title="Engineer", url="https://jobs.example/nav-browse")
    assert _apply_listing(
        nav,
        "<header><h2>Browse jobs</h2></header>"
        "<title>Engineer</title><p>Salary $180,000</p>",
    ) is True
    assert nav.pay_high == 180_000
    copy = Opportunity(title="Engineer", url="https://jobs.example/browse-copy")
    assert _apply_listing(
        copy,
        "<title>Engineer</title>"
        "<p>You can browse jobs on our careers page. Salary $180,000</p>",
    ) is True
    assert copy.pay_high == 180_000
    open_pos = Opportunity(title="Engineer", url="https://jobs.example/open-pos")
    assert _apply_listing(
        open_pos,
        "<title>Engineer</title><p>Great team. Apply now.</p>"
        "<h2>Open positions</h2><p>Account Executive $180,000 - $220,000</p>",
    ) is False
    assert open_pos.pay_high is None
    own_open = Opportunity(title="Engineer", url="https://jobs.example/own-open")
    assert _apply_listing(
        own_open,
        "<title>Engineer</title><p>$115,000 - $145,000 a year.</p>"
        "<h2>Open positions</h2><p>Account Executive $180,000 - $220,000</p>",
    ) is True
    assert own_open.pay_high == 145_000
    rec = Opportunity(title="Engineer", url="https://jobs.example/rec-for-you")
    assert _apply_listing(
        rec,
        "<title>Engineer</title><p>Great team. $180,000</p>"
        "<h2>Recommended for you</h2><p>Account Executive $422,000</p>",
    ) is True
    assert rec.pay_high == 180_000
    company_jobs = Opportunity(title="Engineer", url="https://jobs.example/co-jobs")
    assert _apply_listing(
        company_jobs,
        "<title>Engineer</title><p>$115,000 - $145,000 a year.</p>"
        "<h2>Jobs at this company</h2><p>Account Executive $220,000</p>",
    ) is True
    assert company_jobs.pay_high == 145_000
    more_co = Opportunity(title="Engineer", url="https://jobs.example/more-co")
    assert _apply_listing(
        more_co,
        "<title>Engineer</title><p>$115,000 a year.</p>"
        "<h2>More from this company</h2><p>Account Executive $220,000</p>",
    ) is True
    assert more_co.pay_high == 115_000
    open_copy = Opportunity(title="Engineer", url="https://jobs.example/open-copy")
    assert _apply_listing(
        open_copy,
        "<title>Engineer</title>"
        "<p>We have open positions across the company. Salary $180,000</p>",
    ) is True
    assert open_copy.pay_high == 180_000
    nav_open = Opportunity(title="Engineer", url="https://jobs.example/nav-open")
    assert _apply_listing(
        nav_open,
        "<header><h2>Open positions</h2></header>"
        "<title>Engineer</title><p>Salary $180,000</p>",
    ) is True
    assert nav_open.pay_high == 180_000
    view_all = Opportunity(title="Engineer", url="https://jobs.example/view-all")
    assert _apply_listing(
        view_all,
        "<title>Engineer</title><p>Great team. Apply now.</p>"
        "<h2>View all jobs</h2><p>Account Executive $180,000 - $220,000</p>",
    ) is False
    assert view_all.pay_high is None
    current = Opportunity(title="Engineer", url="https://jobs.example/cur-open")
    assert _apply_listing(
        current,
        "<title>Engineer</title><p>$115,000 - $145,000 a year.</p>"
        "<h2>Current openings</h2><p>Account Executive $220,000</p>",
    ) is True
    assert current.pay_high == 145_000
    open_roles = Opportunity(title="Engineer", url="https://jobs.example/open-roles")
    assert _apply_listing(
        open_roles,
        "<title>Engineer</title><p>Great team. Apply now.</p>"
        "<h2>Open roles</h2><p>Account Executive $180,000 - $220,000</p>",
    ) is False
    assert open_roles.pay_high is None
    roles_copy = Opportunity(title="Engineer", url="https://jobs.example/roles-copy")
    assert _apply_listing(
        roles_copy,
        "<title>Engineer</title>"
        "<p>We have open roles across the company. Salary $180,000</p>",
    ) is True
    assert roles_copy.pay_high == 180_000
    nav_view = Opportunity(title="Engineer", url="https://jobs.example/nav-view")
    assert _apply_listing(
        nav_view,
        "<header><h2>View all jobs</h2></header>"
        "<title>Engineer</title><p>Salary $180,000</p>",
    ) is True
    assert nav_view.pay_high == 180_000
    all_jobs = Opportunity(title="Engineer", url="https://jobs.example/all-jobs")
    assert _apply_listing(
        all_jobs,
        "<title>Engineer</title><p>Great team. Apply now.</p>"
        "<h2>All jobs</h2><p>Account Executive $180,000 - $220,000</p>",
    ) is False
    assert all_jobs.pay_high is None
    jobs_at = Opportunity(title="Engineer", url="https://jobs.example/jobs-at")
    assert _apply_listing(
        jobs_at,
        "<title>Engineer</title><p>$115,000 - $145,000 a year.</p>"
        "<h2>Jobs at Acme</h2><p>Account Executive $220,000</p>",
    ) is True
    assert jobs_at.pay_high == 145_000
    browse_open = Opportunity(title="Engineer", url="https://jobs.example/br-open")
    assert _apply_listing(
        browse_open,
        "<title>Engineer</title><p>Great team. Apply now.</p>"
        "<h2>Browse openings</h2><p>Account Executive $180,000 - $220,000</p>",
    ) is False
    assert browse_open.pay_high is None
    browse_copy = Opportunity(title="Engineer", url="https://jobs.example/br-copy")
    assert _apply_listing(
        browse_copy,
        "<title>Engineer</title>"
        "<p>You can browse openings on our careers page. Salary $180,000</p>",
    ) is True
    assert browse_copy.pay_high == 180_000
    see_all = Opportunity(title="Engineer", url="https://jobs.example/see-all")
    assert _apply_listing(
        see_all,
        "<title>Engineer</title><p>Great team. Apply now.</p>"
        "<h2>See all jobs</h2><p>Account Executive $180,000 - $220,000</p>",
    ) is False
    assert see_all.pay_high is None
    latest_open = Opportunity(title="Engineer", url="https://jobs.example/lat-open")
    assert _apply_listing(
        latest_open,
        "<title>Engineer</title><p>$115,000 - $145,000 a year.</p>"
        "<h2>Latest openings</h2><p>Account Executive $220,000</p>",
    ) is True
    assert latest_open.pay_high == 145_000
    also_applied = Opportunity(title="Engineer", url="https://jobs.example/also-app")
    assert _apply_listing(
        also_applied,
        "<title>Engineer</title><p>Great team. Apply now.</p>"
        "<h2>People also applied</h2><p>Account Executive $180,000 - $220,000</p>",
    ) is False
    assert also_applied.pay_high is None
    explore_c = Opportunity(title="Engineer", url="https://jobs.example/exp-car")
    assert _apply_listing(
        explore_c,
        "<title>Engineer</title>"
        "<p>Explore careers on our site. Salary $180,000</p>",
    ) is True
    assert explore_c.pay_high == 180_000
    more_from = Opportunity(title="Engineer", url="https://jobs.example/more-from")
    assert _apply_listing(
        more_from,
        "<title>Engineer</title><p>$115,000 - $145,000 a year.</p>"
        "<h2>More from Acme</h2><p>Account Executive $220,000</p>",
    ) is True
    assert more_from.pay_high == 145_000
    view_open = Opportunity(title="Engineer", url="https://jobs.example/view-open")
    assert _apply_listing(
        view_open,
        "<title>Engineer</title><p>Great team.</p>"
        "<h2>View all openings</h2><p>Account Executive $220,000</p>",
    ) is False
    assert view_open.pay_high is None
    see_open = Opportunity(title="Engineer", url="https://jobs.example/see-open")
    assert _apply_listing(
        see_open,
        "<title>Engineer</title><p>Great team.</p>"
        "<h2>See all openings</h2><p>Account Executive $220,000</p>",
    ) is False
    assert see_open.pay_high is None
    disc_open = Opportunity(title="Engineer", url="https://jobs.example/disc-open")
    assert _apply_listing(
        disc_open,
        "<title>Engineer</title><p>Great team.</p>"
        "<h2>Discover openings</h2><p>Account Executive $220,000</p>",
    ) is False
    assert disc_open.pay_high is None
    exp_roles = Opportunity(title="Engineer", url="https://jobs.example/exp-roles")
    assert _apply_listing(
        exp_roles,
        "<title>Engineer</title><p>Great team.</p>"
        "<h2>Explore roles</h2><p>Account Executive $220,000</p>",
    ) is False
    assert exp_roles.pay_high is None
    applied_to = Opportunity(title="Engineer", url="https://jobs.example/applied-to")
    assert _apply_listing(
        applied_to,
        "<title>Engineer</title><p>Great team.</p>"
        "<h2>Jobs you applied to</h2><p>Account Executive $220,000</p>",
    ) is False
    assert applied_to.pay_high is None
    view_open_copy = Opportunity(title="Engineer", url="https://jobs.example/vo-copy")
    assert _apply_listing(
        view_open_copy,
        "<title>Engineer</title>"
        "<p>View all openings on our careers page. Salary $180,000</p>",
    ) is True
    assert view_open_copy.pay_high == 180_000
    roles_body = Opportunity(title="Engineer", url="https://jobs.example/roles-body")
    assert _apply_listing(
        roles_body,
        "<title>Engineer</title>"
        "<p>Explore roles across the company. Salary $180,000</p>",
    ) is True
    assert roles_body.pay_high == 180_000
    browse_roles = Opportunity(title="Engineer", url="https://jobs.example/br-roles")
    assert _apply_listing(
        browse_roles,
        "<title>Engineer</title><p>Great team.</p>"
        "<h2>Browse roles</h2><p>Account Executive $220,000</p>",
    ) is False
    assert browse_roles.pay_high is None
    all_open = Opportunity(title="Engineer", url="https://jobs.example/all-open")
    assert _apply_listing(
        all_open,
        "<title>Engineer</title><p>Great team.</p>"
        "<h2>All openings</h2><p>Account Executive $220,000</p>",
    ) is False
    assert all_open.pay_high is None
    view_roles = Opportunity(title="Engineer", url="https://jobs.example/view-roles")
    assert _apply_listing(
        view_roles,
        "<title>Engineer</title><p>Great team.</p>"
        "<h2>View all roles</h2><p>Account Executive $220,000</p>",
    ) is False
    assert view_roles.pay_high is None
    open_jobs = Opportunity(title="Engineer", url="https://jobs.example/open-jobs-h")
    assert _apply_listing(
        open_jobs,
        "<title>Engineer</title><p>Great team.</p>"
        "<h2>Open jobs</h2><p>Account Executive $220,000</p>",
    ) is False
    assert open_jobs.pay_high is None
    browse_roles_copy = Opportunity(title="Engineer", url="https://jobs.example/br-r-copy")
    assert _apply_listing(
        browse_roles_copy,
        "<title>Engineer</title>"
        "<p>Browse roles on our careers page. Salary $180,000</p>",
    ) is True
    assert browse_roles_copy.pay_high == 180_000


def test_apply_listing_ignores_related_jsonld_jobposting():
    from src.engine import _apply_listing

    html = """
    <title>IT Security Administrator at Bitwarden</title>
    <script type="application/ld+json">
    {"@graph": [
      {"@type":"ItemList","itemListElement":[
        {"@type":"JobPosting","title":"Renewals Manager",
         "baseSalary":{"currency":"USD","value":{"minValue":422000,"maxValue":502000,"unitText":"YEAR"}}}
      ]},
      {"@type":"JobPosting","title":"IT Security Administrator",
       "hiringOrganization":{"name":"Bitwarden"},
       "baseSalary":{"currency":"USD","value":{"minValue":115000,"maxValue":145000,"unitText":"YEAR"}}}
    ]}
    </script>
    """
    opp = Opportunity(
        title="IT Security Administrator",
        url="https://wellfound.com/jobs/4335648-it-security-administrator",
    )
    _apply_listing(opp, html)
    assert opp.pay_low == 115_000
    assert opp.pay_high == 145_000
    assert opp.company == "Bitwarden"
    untitled = Opportunity(
        title="IT Security Administrator",
        url="https://wellfound.com/jobs/4335648-it-security-administrator-b",
    )
    no_title = """
    <script type="application/ld+json">
    {"@graph": [
      {"@type":"ItemList","itemListElement":[
        {"@type":"JobPosting","title":"Renewals Manager",
         "baseSalary":{"currency":"USD","value":{"minValue":422000,"maxValue":502000,"unitText":"YEAR"}}}
      ]},
      {"@type":"JobPosting","title":"IT Security Administrator",
       "hiringOrganization":{"name":"Bitwarden"},
       "baseSalary":{"currency":"USD","value":{"minValue":115000,"maxValue":145000,"unitText":"YEAR"}}}
    ]}
    </script>
    <p>Great team. Apply now.</p>
    """
    assert _apply_listing(untitled, no_title) is True
    assert untitled.pay_high == 145_000
    assert untitled.company == "Bitwarden"
    generic = Opportunity(
        title="IT Security Administrator",
        url="https://wellfound.com/jobs/4335648-it-security-administrator-c",
    )
    careers = """
    <title>Careers | Bitwarden</title>
    <script type="application/ld+json">
    {"@graph": [
      {"@type":"ItemList","itemListElement":[
        {"@type":"JobPosting","title":"Renewals Manager",
         "baseSalary":{"currency":"USD","value":{"minValue":422000,"maxValue":502000,"unitText":"YEAR"}}}
      ]},
      {"@type":"JobPosting","title":"IT Security Administrator",
       "hiringOrganization":{"name":"Bitwarden"},
       "baseSalary":{"currency":"USD","value":{"minValue":115000,"maxValue":145000,"unitText":"YEAR"}}}
    ]}
    </script>
    """
    assert _apply_listing(generic, careers) is True
    assert generic.pay_high == 145_000
    details = Opportunity(title="Engineer", url="https://jobs.example/job-details")
    two = """
    <title>Job Details</title>
    <script type="application/ld+json">
    {"@graph": [
      {"@type":"JobPosting","title":"Renewals Manager",
       "baseSalary":{"currency":"USD","value":{"minValue":422000,"maxValue":502000,"unitText":"YEAR"}}},
      {"@type":"JobPosting","title":"Engineer",
       "hiringOrganization":{"name":"Acme"},
       "baseSalary":{"currency":"USD","value":{"minValue":180000,"maxValue":220000,"unitText":"YEAR"}}}
    ]}
    </script>
    """
    assert _apply_listing(details, two) is True
    assert details.pay_high == 220_000
    assert details.company == "Acme"
    solo = Opportunity(title="Engineer", url="https://jobs.example/only-list")
    only_list = """
    <title>Engineer at Acme</title>
    <script type="application/ld+json">
    {"@type":"ItemList","itemListElement":[
      {"@type":"JobPosting","title":"Engineer",
       "hiringOrganization":{"name":"Acme"},
       "baseSalary":{"currency":"USD","value":{"minValue":180000,"maxValue":220000,"unitText":"YEAR"}}}
    ]}
    </script>
    """
    assert _apply_listing(solo, only_list) is True
    assert solo.pay_high == 220_000
    assert solo.company == "Acme"


def test_apply_listing_prefers_longer_jsonld_title_over_related_substring():
    from src.engine import _apply_listing

    html = """
    <title>Staff Software Engineer at Acme | Careers</title>
    <script type="application/ld+json">
    {"@graph": [
      {"@type":"JobPosting","title":"Software Engineer",
       "baseSalary":{"currency":"USD","value":{"minValue":400000,"maxValue":500000,"unitText":"YEAR"}}},
      {"@type":"JobPosting","title":"Staff Software Engineer",
       "hiringOrganization":{"name":"Acme"},
       "baseSalary":{"currency":"USD","value":{"minValue":180000,"maxValue":220000,"unitText":"YEAR"}}}
    ]}
    </script>
    """
    opp = Opportunity(title="x", url="https://jobs.example/staff")
    _apply_listing(opp, html)
    assert opp.pay_low == 180_000
    assert opp.pay_high == 220_000
    assert opp.company == "Acme"


def test_listing_plain_text_ignores_script_salaries():
    from src.engine import _listing_plain_text, _parse_pay, _visible_text

    html = '<script>budget = "$500,000"</script><p>Apply now. No salary listed.</p>'
    assert _parse_pay(_visible_text(html)) == (None, 500_000)
    assert _parse_pay(_listing_plain_text(html)) == (None, None)


def test_public_http_url_rejects_localhost():
    from src.engine import _public_http_url

    assert _public_http_url("https://careers.example/x") is True
    assert _public_http_url("http://127.0.0.1/secret") is False
    assert _public_http_url("javascript:alert(1)") is False


def test_extract_batch_fills_rows_llm_omitted_and_dedupes_url_aliases():
    engine = Engine()
    engine.openai = _fake_client(
        json.dumps(
            {
                "opportunities": [
                    {
                        "title": "From LLM",
                        "url": "https://jobs.example/a",
                        "pay_high": 200_000,
                        "hours_per_week": 20,
                    },
                    {
                        "title": "Alias of A",
                        "url": "HTTPS://JOBS.EXAMPLE/A/",
                        "pay_high": 1,
                        "hours_per_week": 1,
                    },
                ]
            }
        )
    )
    batch = [
        {"title": "Raw A", "url": "https://jobs.example/a", "description": "", "pay": 90_000, "hours": 40},
        {"title": "Junior Developer", "url": "https://jobs.example/b", "description": "hybrid"},
    ]
    out = asyncio.run(engine._extract_batch(batch, "q"))
    assert [o.url for o in out] == ["https://jobs.example/a", "https://jobs.example/b"]
    assert out[0].title == "From LLM"
    assert out[0].pay_high == 90_000
    assert out[0].hours_per_week == 40
    assert out[1].title == "Junior Developer"
    assert out[1].pay_high is None
    assert out[1].score() == 0
