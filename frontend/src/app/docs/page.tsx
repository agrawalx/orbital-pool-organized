'use client'
import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../../components/ui/Card';
import OrbitalInvariant from "./OrbitalInvariant";
import { BlockMath } from 'react-katex';
import {
  Book,
  Code,
  Zap,
  Shield,
  ChevronRight,
  Copy,
  Check,
  FileText,
  Terminal,
  Cpu,
  Network,
  Eye
} from 'lucide-react';
import Navigation from '../../components/Navigation';

const DocumentationPage = () => {
  const [copiedCode, setCopiedCode] = useState<string | null>(null);

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedCode(id);
    setTimeout(() => setCopiedCode(null), 2000);
  };

  const scrollToSection = (sectionId: string) => {
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({
        behavior: 'smooth',
        block: 'start'
      });
    }
  };

  const CodeBlock = ({ code, language, id }: { code: string; language: string; id: string }) => (
    <div className="relative bg-neutral-900 rounded-lg border border-neutral-800">
      <div className="flex items-center justify-between px-4 py-2 border-b border-neutral-800">
        <span className="text-xs text-neutral-400 font-mono">{language}</span>
        <button
          onClick={() => copyToClipboard(code, id)}
          className="flex items-center space-x-1 text-xs text-neutral-400 hover:text-blue-400 transition-colors"
        >
          {copiedCode === id ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
          <span>{copiedCode === id ? 'Copied' : 'Copy'}</span>
        </button>
      </div>
      <pre className="p-4 overflow-x-auto text-sm">
        <code className="text-neutral-300">{code}</code>
      </pre>
    </div>
  );

  return (
    <div className="min-h-screen bg-black text-white">
      <Navigation />

      <div className="container mx-auto px-6 py-24">
        <div className="text-center mb-16">
          <div className="flex items-center justify-center space-x-3 mb-6">
            <div className="p-4 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-2xl">
              <Book className="w-10 h-10 text-white" />
            </div>
            <h1 className="text-5xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
              Orbital Documentation
            </h1>
          </div>
          <p className="text-neutral-400 text-xl max-w-3xl mx-auto leading-relaxed">
            This page provides a comprehensive documentation for the Orbital Protocol, a revolutionary multi-stablecoin AMM using spherical geometry and torus-based invariants for optimal capital efficiency.
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-8 mb-16">
          <Card
            className="bg-neutral-900/50 border-neutral-800 hover:border-blue-500/50 transition-all duration-300 cursor-pointer group"
            onClick={() => scrollToSection('protocol')}
          >
            <CardContent className="p-8">
              <div className="flex items-center space-x-4 mb-4">
                <div className="p-3 bg-blue-500/10 rounded-lg group-hover:bg-blue-500/20 transition-colors">
                  <FileText className="w-8 h-8 text-blue-400" />
                </div>
                <div>
                  <h3 className="text-2xl font-semibold">Core Protocol Documentation</h3>
                  <p className="text-neutral-400">Learn about Orbital&apos;s revolutionary spherical AMM design</p>
                </div>
              </div>
              <div className="flex items-center text-blue-400 text-sm group-hover:text-cyan-300 transition-colors">
                <span>Explore Protocol</span>
                <ChevronRight className="w-4 h-4 ml-1" />
              </div>
            </CardContent>
          </Card>

          <Card
            className="bg-neutral-900/50 border-neutral-800 hover:border-blue-500/50 transition-all duration-300 cursor-pointer group"
            onClick={() => scrollToSection('api')}
          >
            <CardContent className="p-8">
              <div className="flex items-center space-x-4 mb-4">
                <div className="p-3 bg-blue-500/10 rounded-lg group-hover:bg-blue-500/20 transition-colors">
                  <Terminal className="w-8 h-8 text-blue-400" />
                </div>
                <div>
                  <h3 className="text-2xl font-semibold">API Documentation</h3>
                  <p className="text-neutral-400">Complete REST API reference and integration guide</p>
                </div>
              </div>
              <div className="flex items-center text-blue-400 text-sm group-hover:text-cyan-300 transition-colors">
                <span>View API Docs</span>
                <ChevronRight className="w-4 h-4 ml-1" />
              </div>
            </CardContent>
          </Card>
        </div>

        <section id="protocol" className="mb-20">
          <Card className="bg-neutral-900/30 border-neutral-800">
            <CardHeader>
              <CardTitle className="text-3xl flex items-center space-x-3">
                <FileText className="w-8 h-8 text-blue-400" />
                <span>Core Protocol Documentation</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-8">
              <div>
                <h3 className="text-2xl font-semibold mb-4 text-blue-400">Orbital Protocol Overview</h3>
                <p className="text-neutral-300 leading-relaxed mb-6">
                  Orbital is an innovative Automated Market Maker (AMM), an implementation based on the Paradigm&apos;s whitepaper, that transforms stablecoin trading by incorporating a torus-based invariant at the core of its mechanism. While other AMMs use constant product or similar formulas, Orbital positions token reserves as vectors along an n-dimensional space to allow for exact and effective price discovery. This method enables concentrated liquidity customization across pools of stablecoins, with liquidity centered around the $1 equal price point. By coming to a set of tick boundaries as orbits around this middle point, Orbital provides greater capital efficiency and flexibility to liquidity providers and a new benchmark in decentralized finance.
                </p>

                <div className="bg-neutral-800/30 border border-blue-500/30 rounded-lg p-6">
                  <h4 className="font-semibold mb-3 text-cyan-300 flex items-center">
                    <Zap className="w-5 h-5 mr-2" />
                    Key Innovations
                  </h4>
                  <p className="text-neutral-300 text-sm mb-4">
                    Orbital prices assets on a sphere rather than a line, using the invariant <code className=" rounded text-cyan-300"><OrbitalInvariant /></code>
                    <p>where:</p>
                    <ul className='px-8 mt-2'>
                      <li><strong>x<sub>i</sub></strong>: reserve of token i</li>
                      <li><strong>r</strong>: constant radius of the torus</li>
                      <li><strong>n</strong>: total number of tokens in the pool</li>
                    </ul>
                  </p>
                  <div className="grid md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <div className="flex items-center text-sm">
                        <span className="w-2 h-2 bg-cyan-400 rounded-full mr-3"></span>
                        <span className="text-neutral-300">Depeg Tolerance for LPs</span>
                      </div>
                      <div className="flex items-center text-sm">
                        <span className="w-2 h-2 bg-cyan-400 rounded-full mr-3"></span>
                        <span className="text-neutral-300">Q96X48 arithmetic precision</span>
                      </div>
                    </div>
                    <div className="space-y-2">
                      <div className="flex items-center text-sm">
                        <span className="w-2 h-2 bg-cyan-400 rounded-full mr-3"></span>
                        <span className="text-neutral-300">Newton&apos;s method for numerical stability</span>
                      </div>
                      <div className="flex items-center text-sm">
                        <span className="w-2 h-2 bg-cyan-400 rounded-full mr-3"></span>
                        <span className="text-neutral-300">Trade Segmentation in large swaps</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div>
                <h3 className="text-2xl font-semibold mb-4 text-blue-400">Mathematical Foundation</h3>
                <div className="grid md:grid-cols-2 gap-6">
                  <div className="bg-neutral-800/30 rounded-lg p-6">
                    <h4 className="font-semibold mb-3 flex items-center">
                      <Eye className="w-5 h-5 mr-2 text-blue-400" />
                      Torus Invariant
                    </h4>
                    <p className="text-neutral-400 text-sm mb-4">
                      The core mathematical principle behind Orbital&apos;s price discovery mechanism.
                    </p>
                    <div
                      style={{
                        background: '#',
                        padding: '1rem',
                        borderRadius: '8px',
                        display: 'inline-block',
                      }}
                    >
                      <BlockMath math="\sum_{i=1}^{n} (r - x_i)^2 = r^2" />
                    </div>

                  </div>

                  <div className="bg-neutral-800/30 rounded-lg p-6">
                    <h4 className="font-semibold mb-3 flex items-center">
                      <Network className="w-5 h-5 mr-2 text-cyan-400" />
                      Tick Classification
                    </h4>
                    <p className="text-neutral-400 text-sm mb-4">
                      Liquidity positions are classified based on constraint satisfaction.
                    </p>
                    <div className="space-y-3 text-sm">
                      <div className="bg-neutral-700/50 rounded p-3">
                        <div className="flex items-center mb-2">
                          <span className="w-2 h-2 bg-green-400 rounded-full mr-2"></span>
                          <span className="text-green-400 font-medium">Interior Ticks</span>
                        </div>
                      </div>
                      <div className="bg-neutral-700/50 rounded p-3">
                        <div className="flex items-center mb-2">
                          <span className="w-2 h-2 bg-yellow-400 rounded-full mr-2"></span>
                          <span className="text-yellow-400 font-medium">Boundary Ticks</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div>
                <h3 className="text-2xl font-semibold mb-4 text-blue-400">Smart Contract Architecture</h3>
                <div className="space-y-6">
                  {[
                    {
                      title: 'Orbital.sol',
                      icon: Shield,
                      description: 'Main AMM contract implementing the n-token pool with torus invariant mathematics',
                      features: [
                        'Interior/Boundary tick system for liquidity management',
                        'Integration with Stylus math helper for complex calculations',
                        'Swap functionality',
                        'Q96X48 arithmetic with precision',
                        'Depeg Tolerance for liquidity providers'
                      ]
                    },
                    {
                      title: 'OrbitalHelper.rs',
                      icon: Cpu,
                      color: 'cyan',
                      description: 'Stylus contract for mathematical computations',
                      features: [
                        'Torus invariant solving using Newton&apos;s method',
                        'Radius calculations for tick positioning',
                        'Boundary tick S value calculations',
                        'Robust numerical methods with multiple fallbacks',
                        'Integer square root calculations',
                        'Q96X48 calculations'
                      ]
                    },
                  ].map((contract) => (
                    <div key={contract.title} className="bg-neutral-800/30 rounded-lg p-6 border border-neutral-700">
                      <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center space-x-3">
                          <div className={`p-2 rounded-lg ${contract.color === 'blue' ? 'bg-blue-500/10' :
                            contract.color === 'cyan' ? 'bg-cyan-500/10' : 'bg-green-500/10'
                            }`}>
                            <contract.icon className={`w-6 h-6 ${contract.color === 'blue' ? 'text-blue-400' :
                              contract.color === 'cyan' ? 'text-cyan-400' : 'text-green-400'
                              }`} />
                          </div>
                          <div>
                            <h4 className="text-lg font-semibold">{contract.title}</h4>
                            <p className="text-neutral-400 text-sm">{contract.description}</p>
                          </div>
                        </div>
                      </div>

                      <div className="grid md:grid-cols-2 gap-3">
                        {contract.features.map((feature, idx) => (
                          <div key={idx} className="flex items-start text-sm">
                            <span className={`w-1.5 h-1.5 rounded-full mr-3 mt-2 flex-shrink-0 ${contract.color === 'blue' ? 'bg-blue-400' :
                              contract.color === 'cyan' ? 'bg-cyan-400' : 'bg-green-400'
                              }`}></span>
                            <span className="text-neutral-300">{feature}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <h3 className="text-2xl font-semibold mb-4 text-blue-400">Network & Deployment</h3>
                <div className="bg-neutral-800/30 rounded-lg p-6">
                  <div className="grid md:grid-cols-2 gap-6">
                    <div>
                      <h4 className="font-semibold mb-3 text-cyan-300">Network Details</h4>
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-neutral-400">Network:</span>
                          <span className="text-white">Arbitrum Sepolia</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-neutral-400">Chain ID:</span>
                          <span className="text-white">421614</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-neutral-400">RPC URL:</span>
                          <span className="text-white text-xs">sepolia-rollup.arbitrum.io/rpc</span>
                        </div>
                      </div>
                    </div>
                    <div>
                      <h4 className="font-semibold mb-3 text-cyan-300">Contract Addresses</h4>
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-neutral-400">Pool:</span>
                          <code className="text-white text-xs bg-neutral-700 px-2 py-1 rounded">0x83EC...6431</code>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-neutral-400">Tokens:</span>
                          <span className="text-white">MUSDC-A to MUSDC-E</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-neutral-400">API:</span>
                          <code className="text-white text-xs bg-neutral-700 px-2 py-1 rounded">orbital-pool-api.onrender.com</code>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </section>

        <section id="api" className="mb-20">
          <Card className="bg-neutral-900/30 border-neutral-800">
            <CardHeader>
              <CardTitle className="text-3xl flex items-center space-x-3">
                <Terminal className="w-8 h-8 text-blue-400" />
                <span>API Documentation</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-8">
              <div>
                <h3 className="text-2xl font-semibold mb-4 text-blue-400">Base URL & Setup</h3>
                <div className="bg-gradient-to-r from-blue-500/10 to-cyan-500/10 border border-blue-500/30 rounded-lg p-6">
                  <CodeBlock
                    id="base-url"
                    language="endpoint"
                    code="http://orbital-pool-api.onrender.com"
                  />
                  <p className="text-neutral-300 text-sm mt-4 mb-4">
                    All API endpoints are accessible at this base URL. The API is optimized for command-line testing with CURL commands.
                  </p>
                  <div className="grid md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <div className="flex items-center text-sm">
                        <span className="w-2 h-2 bg-blue-400 rounded-full mr-3"></span>
                        <span className="text-neutral-300">RESTful API design</span>
                      </div>
                      <div className="flex items-center text-sm">
                        <span className="w-2 h-2 bg-cyan-400 rounded-full mr-3"></span>
                        <span className="text-neutral-300">JSON request/response format</span>
                      </div>
                    </div>
                    <div className="space-y-2">
                      <div className="flex items-center text-sm">
                        <span className="w-2 h-2 bg-blue-400 rounded-full mr-3"></span>
                        <span className="text-neutral-300">Command-line optimized</span>
                      </div>
                      <div className="flex items-center text-sm">
                        <span className="w-2 h-2 bg-cyan-400 rounded-full mr-3"></span>
                        <span className="text-neutral-300">All amounts in wei (18 decimals)</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div>
                <h3 className="text-2xl font-semibold mb-6 text-blue-400">Core Endpoints</h3>
                <div className="space-y-6">
                  {[
                    {
                      method: 'GET',
                      endpoint: '/health',
                      desc: 'Check API and pool status - returns pool health, contract addresses, and network info',
                      example: 'curl http://orbital-pool-api.onrender.com/health',
                      response: `{
  "status": "healthy",
  "pool_address": "0x83EC719A6F504583d0F88CEd111cB8e8c0956431",
  "network": "arbitrum_sepolia",
  "chain_id": 421614
}`
                    },
                    {
                      method: 'GET',
                      endpoint: '/tokens',
                      desc: 'Get all supported tokens (MUSDC-A through MUSDC-E) with addresses and metadata',
                      example: 'curl http://orbital-pool-api.onrender.com/tokens',
                      response: `{
  "tokens": [
    {"index": 0, "symbol": "MUSDC-A", "address": "0x..."},
    {"index": 1, "symbol": "MUSDC-B", "address": "0x..."},
    ...
  ]
}`
                    },
                    {
                      method: 'POST',
                      endpoint: '/swap',
                      desc: 'Execute token swap - returns transaction data for wallet signing',
                      example: `curl -X POST http://orbital-pool-api.onrender.com/swap \\
  -H "Content-Type: application/json" \\
  -d '{
    "token_in_index": 0,
    "token_out_index": 1,
    "amount_in": "1000000000000000000",
    "min_amount_out": "0",
    "user_address": "0xYourAddress"
  }'`,
                      response: `{
  "success": true,
  "transaction_data": {...},
  "estimated_output": "999500000000000000",
  "price_impact": "0.05%"
}`
                    },
                    {
                      method: 'POST',
                      endpoint: '/liquidity/add',
                      desc: 'Add liquidity to the pool - specify amounts for all 5 tokens and tolerance percentage',
                      example: `curl -X POST http://orbital-pool-api.onrender.com/liquidity/add \\
  -H "Content-Type: application/json" \\
  -d '{
    "amounts": ["1000000000000000000", "1000000000000000000", "1000000000000000000", "1000000000000000000", "1000000000000000000"],
    "tolerance": 0.5,
    "user_address": "0xYourAddress"
  }'`,
                      response: `{
  "success": true,
  "transaction_data": {...},
  "lp_tokens_minted": "5000000000000000000"
}`
                    },
                    {
                      method: 'POST',
                      endpoint: '/liquidity/remove',
                      desc: 'Remove liquidity from the pool - specify LP token amount to burn',
                      example: `curl -X POST http://orbital-pool-api.onrender.com/liquidity/remove \\
  -H "Content-Type: application/json" \\
  -d '{
    "lp_token_amount": "1000000000000000000",
    "user_address": "0xYourAddress"
  }'`,
                      response: `{
  "success": true,
  "transaction_data": {...},
  "tokens_returned": ["200000000000000000", ...]
}`
                    },
                    {
                      method: 'GET',
                      endpoint: '/gas-price',
                      desc: 'Get current gas price estimation for Arbitrum Sepolia network',
                      example: 'curl http://orbital-pool-api.onrender.com/gas-price',
                      response: `{
  "gas_price": "100000000",
  "gas_price_gwei": "0.1",
  "network": "arbitrum_sepolia"
}`
                    }
                  ].map((endpoint, index) => (
                    <div key={index} className="bg-neutral-800/30 rounded-lg p-6 border border-neutral-700">
                      <div className="flex items-center space-x-3 mb-4">
                        <span className={`px-3 py-1 rounded text-xs font-mono font-semibold ${endpoint.method === 'GET' ? 'bg-green-500/20 text-green-400' : 'bg-blue-500/20 text-blue-400'
                          }`}>
                          {endpoint.method}
                        </span>
                        <code className="text-cyan-300 font-mono text-lg">{endpoint.endpoint}</code>
                      </div>
                      <p className="text-neutral-300 text-sm mb-4">{endpoint.desc}</p>

                      <div className="space-y-4">
                        <div>
                          <h5 className="text-sm font-semibold text-neutral-400 mb-2">Request Example:</h5>
                          <CodeBlock
                            id={`endpoint-${index}`}
                            language="bash"
                            code={endpoint.example}
                          />
                        </div>

                        {endpoint.response && (
                          <div>
                            <h5 className="text-sm font-semibold text-neutral-400 mb-2">Response Example:</h5>
                            <CodeBlock
                              id={`response-${index}`}
                              language="json"
                              code={endpoint.response}
                            />
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <h3 className="text-2xl font-semibold mb-4 text-blue-400">Integration Notes</h3>
                <div className="bg-neutral-800/30 rounded-lg p-6">
                  <div className="grid md:grid-cols-2 gap-6">
                    <div>
                      <h4 className="font-semibold mb-3 text-cyan-300 flex items-center">
                        <Shield className="w-5 h-5 mr-2" />
                        Important Requirements
                      </h4>
                      <div className="space-y-2 text-sm">
                        <div className="flex items-start">
                          <span className="w-2 h-2 bg-blue-400 rounded-full mr-3 mt-2 flex-shrink-0"></span>
                          <span className="text-neutral-300"><strong>Network:</strong> Arbitrum Sepolia testnet (Chain ID: 421614)</span>
                        </div>
                        <div className="flex items-start">
                          <span className="w-2 h-2 bg-cyan-400 rounded-full mr-3 mt-2 flex-shrink-0"></span>
                          <span className="text-neutral-300"><strong>Amounts:</strong> All token amounts must be in wei (18 decimals)</span>
                        </div>
                        <div className="flex items-start">
                          <span className="w-2 h-2 bg-blue-400 rounded-full mr-3 mt-2 flex-shrink-0"></span>
                          <span className="text-neutral-300"><strong>Approvals:</strong> Ensure tokens are approved before swapping</span>
                        </div>
                        <div className="flex items-start">
                          <span className="w-2 h-2 bg-cyan-400 rounded-full mr-3 mt-2 flex-shrink-0"></span>
                          <span className="text-neutral-300"><strong>Testing:</strong> Optimized for command-line/curl testing</span>
                        </div>
                      </div>
                    </div>

                    <div>
                      <h4 className="font-semibold mb-3 text-cyan-300 flex items-center">
                        <Code className="w-5 h-5 mr-2" />
                        Quick Integration
                      </h4>
                      <CodeBlock
                        id="quick-integration"
                        language="javascript"
                        code={`// Basic swap example
const response = await fetch('http://orbital-pool-api.onrender.com/swap', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    token_in_index: 0,
    token_out_index: 1,
    amount_in: ethers.parseEther("1").toString(),
    min_amount_out: "0",
    user_address: userAddress
  })
});

const data = await response.json();
if (data.success) {
  // Sign and send transaction
  const tx = await signer.sendTransaction(data.transaction_data);
  await tx.wait();
}`}
                      />
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </section>

      </div>
    </div>
  );
};

export default DocumentationPage;
