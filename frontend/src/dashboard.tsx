import React, { useState, useRef, useEffect } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent } from "@/components/ui/card";
import Papa from 'papaparse';

// Helper function to safely parse JSON-like strings (handles single quotes)
const parseJsonStringField = (jsonString: string | unknown): any[] | any | null => {
  if (typeof jsonString !== 'string' || !jsonString.trim()) {
    return null;
  }
  try {
    // Replace single quotes with double quotes for valid JSON, being careful about escaped quotes
    const validJsonString = jsonString
      .replace(/\'/g, "\\'") // Temporarily protect escaped single quotes
      .replace(/'/g, '"')      // Replace unescaped single quotes
      .replace(/\\'/g, "'");   // Restore protected escaped single quotes
    return JSON.parse(validJsonString);
  } catch (error) {
    console.warn("Failed to parse JSON string field:", jsonString, error);
    return null; // Or return the original string, or a specific error indicator
  }
};

// Helper function to format mineral resources
const formatMineralResources = (resourcesValue: string | number | null | undefined): string => {
  if (!resourcesValue) return 'N/A';
  if (typeof resourcesValue === 'number') return resourcesValue.toLocaleString(); // Should not happen if it's JSON string
  if (typeof resourcesValue !== 'string') return 'Invalid data format';

  const parsed = parseJsonStringField(resourcesValue);
  if (!Array.isArray(parsed)) {
    return resourcesValue; // Return original string if not parsable as an array
  }

  return parsed.map(res => {
    let details = `${res.category || 'N/A'}: ${res.contained_amount?.toLocaleString() || 'N/A'} ${res.contained_unit || ''} ${res.mineral_type || ''}`;
    if (res.grade) {
      details += ` (${res.grade?.toLocaleString()} ${res.grade_unit || ''})`;
    }
    if (res.additional_minerals && typeof res.additional_minerals === 'object') {
        const additional = Object.entries(res.additional_minerals)
            .map(([key, val]: [string, any]) => `${key}: ${val.contained_amount?.toLocaleString() || 'N/A'} ${val.contained_unit || ''} (${val.grade?.toLocaleString()} ${val.unit || ''})`)
            .join('; ');
        if (additional) details += ` (Additional: ${additional})`;
    }
    return details;
  }).join('; \n') || 'N/A';
};

// Helper function to format average annual production
const formatAverageProduction = (productionValue: string | number | null | undefined): string => {
  if (!productionValue) return 'N/A';
  if (typeof productionValue === 'number') return productionValue.toLocaleString(); // Should not happen if it's JSON string
  if (typeof productionValue !== 'string') return 'Invalid data format';

  const parsed = parseJsonStringField(productionValue);
  if (!Array.isArray(parsed)) {
    return productionValue; // Return original string if not parsable as an array
  }

  return parsed.map(prod => 
    `${prod.mineral || 'N/A'}: ${prod.amount?.toLocaleString() || 'N/A'} ${prod.unit || ''}`
  ).join('; \n') || 'N/A';
};

interface CompanyData {
  report_title?: string | null;
  report_type?: string | null;
  company_name?: string | null;
  project_name?: string | null;
  project_location?: string | null;
  effective_date?: string | null;
  report_date?: string | null;
  prepared_for?: string | null;
  prepared_by?: string | null;
  qualified_persons?: string | null;
  executive_summary?: string | null;
  key_conclusions?: string | null;
  key_recommendations?: string | null;
  mineral_resources?: string | number | null;
  mineral_reserves?: string | number | null;
  study_base_case_description?: string | null;
  currency_used?: string | null;
  discount_rate_percent?: number | null;
  npv_pre_tax_usd?: number | null;
  npv_post_tax_usd?: number | null;
  irr_pre_tax_percent?: number | null;
  irr_post_tax_percent?: number | null;
  payback_period_years_post_tax?: number | null;
  initial_capital_cost_usd?: number | null;
  sustaining_capital_cost_usd?: number | null;
  closure_reclamation_cost_usd?: number | null;
  life_of_mine_years?: number | null;
  mining_method?: string | null;
  processing_method?: string | null;
  average_annual_throughput_tonnes?: number | null;
  average_annual_production?: string | number | null;
  total_operating_cost_per_tonne_usd?: number | null;
  commodity_price_assumptions?: string | null;
  economic_sensitivities?: string | null;
  infrastructure_summary?: string | null;
  environmental_social_governance_summary?: string | null;
  risks_and_opportunities?: string | null;
  ticker?: string | null;
  "average_recovery_percent.gold"?: number | null;
  "operating_cost_summary.LHOS_Stope_Production_usd_per_tonne"?: number | null;
  "operating_cost_summary.LHOS_Operating_Development_usd_per_tonne"?: number | null;
  "operating_cost_summary.CF_Operating_Development_usd_per_tonne"?: number | null;
  "operating_cost_summary.Ore_Waste_Handling_usd_per_tonne"?: number | null;
  "operating_cost_summary.Services_Ancillary_usd_per_tonne"?: number | null;
  "operating_cost_summary.Maintenance_usd_per_tonne"?: number | null;
  "operating_cost_summary.Supervision_Technical_usd_per_tonne"?: number | null;
  "aisc_usd_per_ounce.gold_oz"?: number | null;
  index?: number | null;
  current_price?: number | null;
  currency?: string | null;
  market_cap?: number | null;
  shares_outstanding?: number | null;
  enterprise_value?: number | null;
  beta?: number | null;
  trailing_pe?: number | null;
  forward_pe?: number | null;
  ReportDate?: string | null;
  "Cash And Cash Equivalents"?: number | null;
  "Total Debt"?: number | null;
  "Current Debt"?: number | null;
  "Tangible Book Value"?: number | null;
  "Working Capital"?: number | null;
  "Net Tangible Assets"?: number | null;
  average_recovery_percent?: number | null;
  operating_cost_summary?: string | null;
  aisc_usd_per_ounce?: number | null;
  "average_recovery_percent.silver"?: number | null;
  "operating_cost_summary.mining_usd_per_tonne"?: number | null;
  "operating_cost_summary.processing_usd_per_tonne"?: number | null;
  "operating_cost_summary.ga_usd_per_tonne"?: number | null;
  "operating_cost_summary.ore_haul_usd_per_tonne"?: number | null;
  commodity?: string | null;
  fileName?: string;
}

const sampleData: CompanyData[] = [
  {
    company_name: 'Alpha Gold',
    npv_post_tax_usd: 500,
    market_cap: 250,
    report_type: 'Development',
    commodity: 'Gold',
    irr_post_tax_percent: 25,
    "aisc_usd_per_ounce.gold_oz": 900,
    mineral_resources: "2.5 Moz",
    average_annual_production: "150 koz/y",
    life_of_mine_years: 12,
  },
  {
    company_name: 'Beta Minerals',
    npv_post_tax_usd: 300,
    market_cap: 350,
    report_type: 'Production',
    commodity: 'Copper',
    irr_post_tax_percent: 18,
    "aisc_usd_per_ounce.gold_oz": 1200,
    mineral_resources: "1.8 Moz",
    average_annual_production: "100 koz/y",
    life_of_mine_years: 8,
  },
  {
    company_name: 'Gamma Metals',
    npv_post_tax_usd: 800,
    market_cap: 400,
    report_type: 'Exploration',
    commodity: 'Lithium',
    irr_post_tax_percent: 30,
    "aisc_usd_per_ounce.gold_oz": 700,
    mineral_resources: "3.2 Moz",
    average_annual_production: "180 koz/y",
    life_of_mine_years: 15,
  },
];

// New function to process raw CSV text into CompanyData[]
const processCsvText = async (csvText: string, fileName?: string): Promise<CompanyData[]> => {
  return new Promise((resolve, reject) => {
    Papa.parse<any>(csvText, { // Use <any> for intermediate processing
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: (result) => {
        if (result.errors.length > 0) {
          console.error(`Papaparse errors in ${fileName || 'uploaded file'}:`, result.errors);
          // Decide if partial data is acceptable or if it should reject
        }

        const processedData = result.data.map(row => {
          let derivedCommodity: string | null = null;
          const projectName = typeof row.project_name === 'string' ? row.project_name.toLowerCase() : "";
          const reportTitle = typeof row.report_title === 'string' ? row.report_title.toLowerCase() : "";

          if (projectName.includes("gold") || reportTitle.includes("gold")) derivedCommodity = "Gold";
          else if (projectName.includes("copper") || reportTitle.includes("copper")) derivedCommodity = "Copper";
          else if (projectName.includes("silver") || reportTitle.includes("silver")) derivedCommodity = "Silver";
          else if (projectName.includes("zinc") || reportTitle.includes("zinc")) derivedCommodity = "Zinc";
          else if (projectName.includes("lead") || reportTitle.includes("lead")) derivedCommodity = "Lead";
          else if (projectName.includes("lithium") || reportTitle.includes("lithium")) derivedCommodity = "Lithium";
          else if (projectName.includes("polymetallic") || reportTitle.includes("polymetallic")) derivedCommodity = "Polymetallic";
          
          return { ...row, commodity: derivedCommodity, fileName }; // Add derived commodity and fileName
        });

        const checkedData = processedData.map(row => {
          const companyNameCheck = row.company_name ? true : false;
          const npvCheck = typeof row.npv_post_tax_usd === 'number'; // Still check for logging
          const marketCapCheck = typeof row.market_cap === 'number'; // Still check for logging
          const commodityCheck = row.commodity ? true : false;
          
          // Core validity: must have a company name and a derived commodity.
          // NPV and MarketCap are desirable for the chart but not strictly required for data to be listed in the table.
          const isCoreValid = companyNameCheck && commodityCheck;

          if (!isCoreValid || !npvCheck || !marketCapCheck) { // Log if core is invalid OR if expected chart data is problematic
            console.warn(`Potentially incomplete row in ${fileName || 'uploaded file'}:`, {
              fileName: fileName,
              companyName: row.company_name,
              derived_commodity: row.commodity,
              npv_post_tax_usd: row.npv_post_tax_usd,
              market_cap: row.market_cap,
              coreValid: isCoreValid,
              companyNamePresent: companyNameCheck,
              isCommodityDerivedAndPresent: commodityCheck,
              isNpvNumber: npvCheck, // Keep this for diagnostics
              isMarketCapNumber: marketCapCheck, // Keep this for diagnostics
            });
          }
          // A row is included in the main dataset if it has at least a company name and a derived commodity.
          // The scatter plot will inherently only show points that have the necessary npv and market_cap numbers.
          return { ...row, __isValid: isCoreValid };
        });

        const validData = checkedData.filter(row => row.__isValid).map(row => {
          const { __isValid, ...rest } = row;
          return rest as CompanyData;
        });
        
        console.log(`Processed ${fileName || 'uploaded file'}: Total rows ${result.data.length}, Valid rows ${validData.length}`);
        resolve(validData);
      },
      error: (error: any) => {
        console.error(`Error parsing CSV text from ${fileName || 'uploaded file'}:`, error);
        reject(error);
      }
    });
  });
};

export default function Dashboard() {
  const [data, setData] = useState<CompanyData[]>([]); // Initialize with empty array
  const [isLoading, setIsLoading] = useState<boolean>(true); // Loading state for initial data
  const [error, setError] = useState<string | null>(null); // Error state for initial data
  const [filter, setFilter] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Effect to load default CSVs on mount
  useEffect(() => {
    const loadDefaultData = async () => {
      setIsLoading(true);
      setError(null);
      let allDefaultData: CompanyData[] = [];
      // IMPORTANT: Adjust the path according to where your CSVs are within the `src` directory
      // e.g., if they are in `src/reports_data/`, the path is `/src/reports_data/*_processed_dashboard_data.csv`
      const reportModules = import.meta.glob('/src/reports_data/*_processed_dashboard_data.csv', { as: 'raw' });

      console.log("Found report modules:", Object.keys(reportModules));

      for (const path in reportModules) {
        try {
          console.log("Loading default CSV:", path);
          const csvText = await reportModules[path]();
          const fileName = path.split('/').pop(); // Extract filename
          const parsedData = await processCsvText(csvText, fileName);
          allDefaultData = allDefaultData.concat(parsedData);
        } catch (e) {
          console.error("Error loading or processing default CSV:", path, e);
          setError(prevError => prevError ? `${prevError}; Failed to load ${path}` : `Failed to load ${path}`);
        }
      }
      
      if (allDefaultData.length === 0 && Object.keys(reportModules).length > 0) {
          console.warn("Default CSVs were found and processed, but resulted in no valid data rows. Check console for warnings from processCsvText.");
          // Optionally set sampleData if no default data is loaded and valid
          // setData(sampleData);
      } else if (allDefaultData.length > 0) {
        setData(allDefaultData);
      } else {
        console.log("No default CSVs found or no data extracted. Initializing with sample data or empty as per logic.");
        // setData(sampleData); // Or keep it empty if that's preferred when no files are found
      }
      setIsLoading(false);
    };

    loadDefaultData();
  }, []); // Empty dependency array ensures this runs only on mount


  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      console.log("Attempting to parse manually uploaded file:", file.name);
      try {
        const csvText = await file.text();
        const parsedData = await processCsvText(csvText, file.name);
        
        if (parsedData.length === 0) {
            console.warn("Manually uploaded CSV parsed, but no valid data rows found. Check console for parsing details.");
            // Potentially show a user-friendly message here
        }
        // Decide if you want to replace or append the data
        setData(parsedData); // Current: replaces data
        // setData(prevData => [...prevData, ...parsedData]); // Alternative: appends data

      } catch (e) {
        console.error("Error processing manually uploaded file:", e);
        // Show user-friendly error message
      }
    }
  };

  const handleUploadButtonClick = () => {
    fileInputRef.current?.click();
  };

  const filteredData = data.filter(d => {
    if (d && typeof d.commodity === 'string') {
      return d.commodity.toLowerCase().includes(filter.toLowerCase());
    }
    return false;
  });

  // Log filteredData before rendering
  console.log("Filtered data passed to components:", filteredData);
  console.log("CTGO data for chart:", filteredData.filter(d => d.company_name === "Contango ORE, Inc."));
  // Assuming the company name for CDE is "Coeur Mining, Inc." - please adjust if different
  console.log("CDE data for chart:", filteredData.filter(d => d.company_name === "Coeur Mining, Inc."));

  return (
    <div className="p-4 space-y-6">
      <h1 className="text-2xl font-bold">Mining Stocks Dashboard</h1>

      <Card>
        <CardContent className="space-y-4 pt-6">
          <div className="flex gap-4 items-center">
            <Input
              placeholder="Filter by commodity (e.g. gold, copper)"
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              className="w-64"
            />
            <Input
              type="file"
              ref={fileInputRef}
              onChange={handleFileUpload}
              style={{ display: 'none' }}
              accept=".csv"
            />
            <Button onClick={handleUploadButtonClick}>Upload CSV</Button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardContent>
          <h2 className="text-xl font-semibold mb-2">NPV vs Market Cap</h2>
          <ResponsiveContainer width="100%" height={400}>
            <ScatterChart>
              <CartesianGrid />
              <XAxis type="number" dataKey="npv_post_tax_usd" name="NPV (Post-Tax)" unit="M USD" />
              <YAxis type="number" dataKey="market_cap" name="Market Cap" unit="M USD" />
              <Tooltip 
                cursor={{ strokeDasharray: '3 3' }} 
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload; // The data object for the hovered point
                    return (
                      <div className="bg-white p-2 border border-gray-300 shadow-lg rounded text-sm max-w-xs break-words">
                        <p className="font-bold text-blue-600">{data.company_name || 'N/A'}</p>
                        <p className="text-xs text-gray-500">{data.fileName || 'N/A'}{data.report_title ? ` - ${data.report_title.substring(0,50)}...` : ' - Report N/A'}</p>
                        <p>{`NPV (Post-Tax): $${(data.npv_post_tax_usd || 0).toLocaleString()}M USD`}</p>
                        <p>{`Market Cap: $${(data.market_cap || 0).toLocaleString()}M USD`}</p>
                        {/* You can add more fields from 'data' here if needed */}
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Scatter name="Companies" data={filteredData} fill="#3b82f6" />
            </ScatterChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      <Card>
        <CardContent>
          <h2 className="text-xl font-semibold mb-2">Data Table</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm text-left">
              <thead>
                <tr className="border-b">
                  <th className="p-2">Company</th>
                  <th className="p-2">Commodity</th>
                  <th className="p-2">Stage</th>
                  <th className="p-2">NPV (Post-Tax M USD)</th>
                  <th className="p-2">Market Cap (M USD)</th>
                  <th className="p-2">NPV/MC</th>
                  <th className="p-2">IRR (Post-Tax %)</th>
                  <th className="p-2">AISC (USD/oz Gold Eq.)</th>
                  <th className="p-2">Resource Size</th>
                  <th className="p-2">Avg. Annual Prod.</th>
                  <th className="p-2">Mine Life (yrs)</th>
                </tr>
              </thead>
              <tbody>
                {filteredData.map((d, i) => (
                  <tr key={i} className="border-b hover:bg-gray-50">
                    <td className="p-2 whitespace-pre-wrap">{d.company_name || 'N/A'}</td>
                    <td className="p-2 whitespace-pre-wrap">{d.commodity || 'N/A'}</td>
                    <td className="p-2 whitespace-pre-wrap">{d.report_type || 'N/A'}</td>
                    <td className="p-2 whitespace-pre-wrap">{typeof d.npv_post_tax_usd === 'number' ? d.npv_post_tax_usd.toLocaleString() : 'N/A'}</td>
                    <td className="p-2 whitespace-pre-wrap">{typeof d.market_cap === 'number' ? d.market_cap.toLocaleString() : 'N/A'}</td>
                    <td className="p-2 whitespace-pre-wrap">
                      {(typeof d.npv_post_tax_usd === 'number' && typeof d.market_cap === 'number' && d.market_cap !== 0)
                        ? (d.npv_post_tax_usd / d.market_cap).toFixed(2)
                        : 'N/A'}
                    </td>
                    <td className="p-2 whitespace-pre-wrap">{typeof d.irr_post_tax_percent === 'number' ? d.irr_post_tax_percent.toFixed(1) : 'N/A'}</td>
                    <td className="p-2 whitespace-pre-wrap">{typeof d["aisc_usd_per_ounce.gold_oz"] === 'number' ? d["aisc_usd_per_ounce.gold_oz"].toLocaleString() : (typeof d.aisc_usd_per_ounce === 'number' ? d.aisc_usd_per_ounce.toLocaleString() : 'N/A')}</td>
                    <td className="p-2 whitespace-pre-wrap">{formatMineralResources(d.mineral_resources)}</td>
                    <td className="p-2 whitespace-pre-wrap">{formatAverageProduction(d.average_annual_production)}</td>
                    <td className="p-2 whitespace-pre-wrap">{typeof d.life_of_mine_years === 'number' ? d.life_of_mine_years : 'N/A'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
} 