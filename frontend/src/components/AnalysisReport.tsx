import ReactMarkdown from 'react-markdown';
import { cn } from '@/lib/utils';
import { TrendingUp, TrendingDown, DollarSign, Activity } from 'lucide-react';

interface AnalysisData {
  symbol: string;
  final_report: string;
  micro_analysis?: any;
}

export function AnalysisReport({ data }: { data: AnalysisData }) {
  const reportLower = data.final_report.toLowerCase();
  const isBullish = reportLower.includes("bullish") || reportLower.includes("strong buy");
  const isBearish = reportLower.includes("bearish") || reportLower.includes("sell");

  return (
    <div className="w-full max-w-4xl mx-auto space-y-6 animate-in slide-in-from-bottom-5 duration-700">
      {/* Header Card */}
      <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-6 backdrop-blur-sm">
        <div className="flex items-start justify-between">
          <div>
            <h2 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-indigo-400">
              {data.symbol}
            </h2>
            <p className="text-gray-400 mt-1">AI Investment Analysis</p>
          </div>
          <div className={cn(
            "px-4 py-2 rounded-full flex items-center space-x-2 font-bold",
            isBullish ? "bg-green-500/20 text-green-400 border border-green-500/30" : 
            isBearish ? "bg-red-500/20 text-red-400 border border-red-500/30" :
            "bg-yellow-500/20 text-yellow-400 border border-yellow-500/30"
          )}>
            {isBullish ? <TrendingUp className="w-5 h-5" /> : 
             isBearish ? <TrendingDown className="w-5 h-5" /> : 
             <Activity className="w-5 h-5" />}
            <span>
              {isBullish ? "STRONG BUY / BULLISH" : isBearish ? "SELL / BEARISH" : "NEUTRAL / HOLD"}
            </span>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        
        {/* Left Column: Quick Stats / Micro Analysis */}
        <div className="md:col-span-1 space-y-6">
           <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-6">
              <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4 flex items-center">
                <Activity className="w-4 h-4 mr-2" /> Micro-Model
              </h3>
              <p className="text-sm text-gray-300 leading-relaxed">
                {data.micro_analysis?.result_summary || "Technical indicators have been integrated into the main report."}
              </p>
           </div>
        </div>

        {/* Right Column: Full Report */}
        <div className="md:col-span-2">
           <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-8 prose prose-invert max-w-none">
              <ReactMarkdown 
                components={{
                  h1: ({node, ...props}) => <h1 className="text-2xl font-bold text-blue-200 mt-0 mb-4" {...props} />,
                  h2: ({node, ...props}) => <h2 className="text-xl font-bold text-gray-200 mt-6 mb-3 border-b border-gray-800 pb-2" {...props} />,
                  h3: ({node, ...props}) => <h3 className="text-lg font-semibold text-gray-300 mt-4 mb-2" {...props} />,
                  strong: ({node, ...props}) => <strong className="text-blue-300 font-bold" {...props} />,
                  ul: ({node, ...props}) => <ul className="list-disc pl-5 space-y-1 text-gray-300" {...props} />,
                }}
              >
                {data.final_report}
              </ReactMarkdown>
           </div>
        </div>
      </div>
    </div>
  );
}
