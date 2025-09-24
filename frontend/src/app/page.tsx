'use client'
import React from 'react';
import Navigation from '../components/Navigation';
import FusionTimeline from '../components/FusionTimeline';
import TechnicalOverview from '../components/TechnicalOverview';
import InteractiveDemo from '../components/InteractiveDemo';
import { Zap, Globe, Shield, ChevronDown } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/Card';
import Link from 'next/link';

const NetworkBackground = () => {
  return (
    <div className="fixed inset-0 -z-10">
      <div className="absolute inset-0 bg-gradient-to-br from-black via-neutral-950 to-black" />
      <div className="mesh-gradient-1 absolute inset-0 opacity-40" />
      <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent" />
    </div>
  );
};

const HeroVideoBackground = () => {
  return (
    <div className="absolute inset-0 -z-10">
      <video
        autoPlay
        muted
        loop
        playsInline
        className="absolute left-0 h-[100%] w-[100%] object-cover opacity-98"
      >
        <source src="/bg.mp4" type="video/mp4" />
      </video>

      <div className="absolute inset-0 bg-black/10" />

      <div className="absolute inset-0 bg-gradient-to-br from-black/30 via-neutral-950/10 to-black/30" />
      <div className="mesh-gradient-1 absolute inset-0 opacity-30" />
      <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent" />
    </div>
  );
};

const Landing = () => {
  const stats = [
    { label: 'Stablecoins Supported', value: '1000+', color: 'from-blue-400 to-cyan-400' },
    { label: 'AMM Innovation', value: 'Spherical', color: 'from-cyan-400 to-blue-500' },
    { label: 'More Efficient than Uniswap and Curve', value: 'Multiple Times', color: 'from-blue-500 to-cyan-500' },
  ];

  const features = [
    {
      icon: <Globe className="w-10 h-10" />,
      title: 'Spherical Geometry',
      description: 'Uses spherical invariant where, enabling optimal capital efficiency.',
      gradient: 'from-blue-500 via-cyan-500 to-blue-600',
    },
    {
      icon: <Shield className="w-10 h-10" />,
      title: 'Concentrated Liquidity',
      description: 'Interior and Boundary tick classification allows for concentrated liquidity positions with high capital efficiency.',
      gradient: 'from-cyan-500 via-blue-600 to-cyan-600',
    },
    {
      icon: <Zap className="w-10 h-10" />,
      title: 'Mathematical Precision',
      description: 'Implementation of Q96X48 arithmetic precision ensure precise trading with minimal slippage.',
      gradient: 'from-blue-600 via-cyan-500 to-blue-700',
    }
  ];

  return (
    <div className="min-h-screen text-white overflow-hidden relative">
      <NetworkBackground />

      <Navigation />

      <section className="relative z-10 px-6 py-12 min-h-[90vh] flex flex-col justify-center">
        <HeroVideoBackground />
        <div className="max-w-7xl mx-auto w-full relative flex flex-col items-center justify-center min-h-[80vh]">

          <div className="text-center mb-12 mt-16">
            <h1 className="text-5xl md:text-7xl lg:text-8xl font-bold mb-6 leading-tight">
              <span className="bg-gradient-to-r from-blue-400 via-cyan-400 to-blue-400 bg-clip-text text-transparent">
                ORBITAL AMM
              </span>
              <span className="bg-gradient-to-r from-white via-neutral-100 to-white bg-clip-text text-transparent"> Protocol</span>
            </h1>
            <p className="text-lg md:text-xl text-neutral-300 max-w-3xl mx-auto leading-relaxed">
              Revolutionary Stablecoin AMM using spherical geometry and orbital mechanics. Trade with optimal capital efficiency through mathematical innovation.
            </p>
          </div>

          <div className="w-full max-w-6xl mx-auto">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 md:gap-8">
              {stats.map((stat, index) => (
                <div key={index} className="group text-center p-4 md:p-6 bg-neutral-900/30 backdrop-blur-sm border border-neutral-800/50 rounded-2xl hover:bg-neutral-900/50 hover:border-neutral-700/60 transition-all duration-300 transform">
                  <div className={`text-xl md:text-2xl lg:text-3xl font-black bg-gradient-to-r ${stat.color} bg-clip-text text-transparent mb-2 transition-transform duration-300 font-[family-name:var(--font-unbounded)]`}>
                    {stat.value}
                  </div>
                  <div className="text-neutral-400 text-xs md:text-sm font-medium group-hover:text-neutral-300 transition-colors duration-300 font-[family-name:var(--font-spline-sans-mono)]">{stat.label}</div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2">
          <a href="#features" aria-label="Scroll to features" className="animate-bounce cursor-pointer group">
            <div className="p-3 bg-neutral-900/30 backdrop-blur-sm border border-neutral-800/50 rounded-full group-hover:bg-neutral-900/50 transition-all duration-300">
              <ChevronDown className="w-6 h-6 text-blue-400 group-hover:text-cyan-300" />
            </div>
          </a>
        </div>
      </section>

      <section id="features" className="relative z-10 px-6 py-12">
        <div className="mesh-gradient-2 absolute inset-0 opacity-30"></div>
        <div className="max-w-6xl mx-auto relative">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-black mb-6 bg-gradient-to-r from-white via-neutral-100 to-white bg-clip-text text-transparent font-[family-name:var(--font-unbounded)]">
              Spherical Innovation
            </h2>
            <p className="text-lg text-neutral-400 max-w-3xl mx-auto font-light font-[family-name:var(--font-spline-sans-mono)]">
              Built with advanced mathematics for the next generation of <br /><span className="text-blue-400 font-semibold">AUTOMATED MARKET MAKERS</span>
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <div key={index} className="group">
                <Card className="h-full transition-transform duration-500">
                  <CardHeader>
                    <div className={`w-16 h-16 bg-gradient-to-br ${feature.gradient} rounded-xl flex items-center justify-center mb-6 transition-all duration-500 shadow-lg shadow-blue-500/20`}>
                      {feature.icon}
                    </div>

                    <CardTitle className="text-xl mb-4 bg-gradient-to-r from-white to-neutral-200 bg-clip-text text-transparent group-hover:from-cyan-200 group-hover:to-white transition-all duration-300">
                      {feature.title}
                    </CardTitle>
                  </CardHeader>

                  <CardContent>
                    <p className="text-neutral-400 leading-relaxed text-sm group-hover:text-neutral-300 transition-colors duration-300">
                      {feature.description}
                    </p>
                  </CardContent>
                </Card>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section id="demo" className="relative z-10 px-6">
        <div className="mesh-gradient-1 absolute inset-0 opacity-30"></div>
        <div className="relative">
          <InteractiveDemo />
        </div>
      </section>

      <section id="how-it-works" className="relative z-10 py-12 bg-black">
        <div className="absolute inset-0 bg-gradient-to-br from-blue-900/10 via-cyan-900/5 to-blue-900/10"></div>
        <div className="relative">
          <FusionTimeline />
        </div>
      </section>

      <section id="technical" className="relative z-10 px-6 py-12 bg-gradient-to-br from-black via-blue-950/20 to-black">
        <div className="absolute inset-0 bg-gradient-to-r from-blue-900/10 via-cyan-900/5 to-blue-900/10 opacity-50"></div>
        <div className="relative">
          <TechnicalOverview />
        </div>
      </section>

      <section id="api" className="relative z-10 px-6 py-12 bg-gradient-to-br from-neutral-950 via-black to-neutral-950">
        <div className="absolute inset-0 bg-gradient-to-r from-blue-900/5 via-cyan-900/5 to-blue-900/5 opacity-30"></div>
        <div className="max-w-7xl mx-auto relative">

          <div className="text-center mb-20">
            <div className="flex items-center justify-center space-x-3 mb-6">
              <div className="p-4 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-2xl">
                <Zap className="w-10 h-10 text-white" />
              </div>
              <h2 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
                Developer API
              </h2>
            </div>
            <p className="text-xl text-neutral-300 max-w-4xl mx-auto leading-relaxed">
              Build on top of Orbital AMM with our simple, powerful REST API. No complex SDKs needed - just straightforward HTTP requests that work with any programming language.
            </p>
          </div>

          <div className="mb-16">
            <Card className="bg-gradient-to-r from-blue-500/10 to-cyan-500/10 border-blue-500/30">
              <CardHeader>
                <CardTitle className="text-2xl flex items-center space-x-3 text-cyan-300">
                  <Globe className="w-6 h-6" />
                  <span>What is an API?</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid md:grid-cols-2 gap-8">
                  <div>
                    <h3 className="text-lg font-semibold mb-4 text-white">Think of it like a restaurant menu</h3>
                    <p className="text-neutral-300 leading-relaxed mb-4">
                      When you go to a restaurant, you don&apos;t walk into the kitchen to cook your own food. Instead, you look at the menu,
                      tell the waiter what you want, and they bring it to you. An API works the same way.
                    </p>
                    <p className="text-neutral-300 leading-relaxed">
                      Our API is like a digital menu for the Orbital AMM. You send us a request saying &quot;I want to swap 100 tokens&quot;
                      and we send back the exact instructions your app needs to make that happen.
                    </p>
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold mb-4 text-white">Why use our API?</h3>
                    <ul className="space-y-3 text-neutral-300">
                      <li className="flex items-start">
                        <span className="w-2 h-2 bg-blue-400 rounded-full mr-3 mt-2 flex-shrink-0"></span>
                        <span><strong>No complex math:</strong> We handle all the spherical geometry calculations for you</span>
                      </li>
                      <li className="flex items-start">
                        <span className="w-2 h-2 bg-blue-400 rounded-full mr-3 mt-2 flex-shrink-0"></span>
                        <span><strong>Works everywhere:</strong> Use any programming language - Python, JavaScript, Go, whatever</span>
                      </li>
                      <li className="flex items-start">
                        <span className="w-2 h-2 bg-blue-400 rounded-full mr-3 mt-2 flex-shrink-0"></span>
                        <span><strong>Real-time data:</strong> Get live prices, gas estimates, and pool information instantly</span>
                      </li>
                      <li className="flex items-start">
                        <span className="w-2 h-2 bg-blue-400 rounded-full mr-3 mt-2 flex-shrink-0"></span>
                        <span><strong>Battle-tested:</strong> Every endpoint works with simple cURL commands</span>
                      </li>
                    </ul>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          <div className="mb-16">
            <h3 className="text-3xl font-bold mb-8 text-center text-white">How It Works (Step by Step)</h3>
            <div className="grid md:grid-cols-3 gap-8">
              <Card className="bg-neutral-900/30 border-neutral-800 group hover:border-blue-500/50 transition-all duration-300">
                <CardContent className="p-8 text-center">
                  <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-full flex items-center justify-center mx-auto mb-6 transition-transform duration-300">
                    <span className="text-2xl font-bold text-white">1</span>
                  </div>
                  <h4 className="text-xl font-semibold mb-4 text-cyan-300">You Send a Request</h4>
                  <p className="text-neutral-300 leading-relaxed">
                    Your app sends us a simple message like &quot;I want to swap 100 MUSDC-A for MUSDC-B&quot;
                    using a standard HTTP request (like visiting a website).
                  </p>
                </CardContent>
              </Card>

              <Card className="bg-neutral-900/30 border-neutral-800 group hover:border-blue-500/50 transition-all duration-300">
                <CardContent className="p-8 text-center">
                  <div className="w-16 h-16 bg-gradient-to-br from-cyan-500 to-blue-500 rounded-full flex items-center justify-center mx-auto mb-6 transition-transform duration-300">
                    <span className="text-2xl font-bold text-white">2</span>
                  </div>
                  <h4 className="text-xl font-semibold mb-4 text-cyan-300">We Do the Math</h4>
                  <p className="text-neutral-300 leading-relaxed">
                    Our servers calculate the exact amount you&apos;ll receive using our spherical invariant formula,
                    check gas prices, and prepare the transaction data.
                  </p>
                </CardContent>
              </Card>

              <Card className="bg-neutral-900/30 border-neutral-800 group hover:border-blue-500/50 transition-all duration-300">
                <CardContent className="p-8 text-center">
                  <div className="w-16 h-16 bg-gradient-to-br from-blue-600 to-cyan-600 rounded-full flex items-center justify-center mx-auto mb-6 transition-transform duration-300">
                    <span className="text-2xl font-bold text-white">3</span>
                  </div>
                  <h4 className="text-xl font-semibold mb-4 text-cyan-300">You Get Instructions</h4>
                  <p className="text-neutral-300 leading-relaxed">
                    We send back everything your app needs: transaction data, gas estimates,
                    and confirmation that everything is ready to go.
                  </p>
                </CardContent>
              </Card>
            </div>
          </div>

          <div className="mb-16">
            <h3 className="text-3xl font-bold mb-8 text-center text-white">What You Can Build</h3>
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
              {[
                {
                  title: "Trading Bots",
                  description: "Create automated trading strategies that execute swaps based on market conditions, price movements, or custom algorithms.",
                  icon: "🤖",
                  color: "from-blue-500 to-cyan-500"
                },
                {
                  title: "Portfolio Managers",
                  description: "Build tools that help users manage their token portfolios, rebalance automatically, and track performance across multiple tokens.",
                  icon: "📊",
                  color: "from-cyan-500 to-blue-500"
                },
                {
                  title: "Analytics Dashboards",
                  description: "Create real-time dashboards showing trading volume, liquidity changes, price charts, and protocol statistics.",
                  icon: "📈",
                  color: "from-blue-600 to-cyan-600"
                },
                {
                  title: "Mobile Apps",
                  description: "Develop mobile applications that let users trade, add liquidity, and monitor their positions on the go.",
                  icon: "📱",
                  color: "from-cyan-600 to-blue-600"
                },
                {
                  title: "DeFi Aggregators",
                  description: "Integrate Orbital AMM into larger DeFi platforms that compare prices across multiple exchanges and find the best deals.",
                  icon: "🔍",
                  color: "from-blue-700 to-cyan-700"
                },
                {
                  title: "Educational Tools",
                  description: "Build learning platforms that teach users about AMMs, liquidity provision, and DeFi concepts through interactive examples.",
                  icon: "🎓",
                  color: "from-cyan-700 to-blue-700"
                }
              ].map((capability, index) => (
                <Card key={index} className="bg-neutral-900/20 border-neutral-800 group hover:border-blue-500/50 transition-all duration-300">
                  <CardContent className="p-6">
                    <div className="flex items-center space-x-4 mb-4">
                      <div className={`w-12 h-12 bg-gradient-to-br ${capability.color} rounded-lg flex items-center justify-center text-2xl`}>
                        {capability.icon}
                      </div>
                      <h4 className="text-lg font-semibold text-white group-hover:text-cyan-300 transition-colors duration-300">
                        {capability.title}
                      </h4>
                    </div>
                    <p className="text-neutral-300 leading-relaxed text-sm">
                      {capability.description}
                    </p>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>

          <div className="mb-16">
            <Card className="bg-gradient-to-r from-cyan-500/10 to-blue-500/10 border-cyan-500/30">
              <CardHeader>
                <CardTitle className="text-2xl text-center text-cyan-300">Why Our API is Different</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid md:grid-cols-2 gap-8">
                  <div>
                    <h4 className="text-lg font-semibold mb-4 text-white">Traditional AMM APIs</h4>
                    <ul className="space-y-2 text-neutral-300">
                      <li className="flex items-start">
                        <span className="text-red-400 mr-2">✗</span>
                        <span>Require complex SDKs and deep blockchain knowledge</span>
                      </li>
                      <li className="flex items-start">
                        <span className="text-red-400 mr-2">✗</span>
                        <span>Only work with specific programming languages</span>
                      </li>
                      <li className="flex items-start">
                        <span className="text-red-400 mr-2">✗</span>
                        <span>Limited to basic swap operations</span>
                      </li>
                      <li className="flex items-start">
                        <span className="text-red-400 mr-2">✗</span>
                        <span>Poor documentation and examples</span>
                      </li>
                    </ul>
                  </div>
                  <div>
                    <h4 className="text-lg font-semibold mb-4 text-white">Orbital AMM API</h4>
                    <ul className="space-y-2 text-neutral-300">
                      <li className="flex items-start">
                        <span className="text-green-400 mr-2">✓</span>
                        <span>Works with any programming language - just HTTP requests</span>
                      </li>
                      <li className="flex items-start">
                        <span className="text-green-400 mr-2">✓</span>
                        <span>Handles complex spherical math automatically</span>
                      </li>
                      <li className="flex items-start">
                        <span className="text-green-400 mr-2">✓</span>
                        <span>Complete trading, liquidity, and analytics in one API</span>
                      </li>
                      <li className="flex items-start">
                        <span className="text-green-400 mr-2">✓</span>
                        <span>Every endpoint tested with real cURL commands</span>
                      </li>
                    </ul>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          <div className="text-center">
            <h3 className="text-3xl font-bold mb-6 text-white">Ready to Start Building?</h3>
            <p className="text-xl text-neutral-300 mb-8 max-w-3xl mx-auto">
              Our API is designed to be simple enough for beginners but powerful enough for advanced developers.
              No complex setup, no confusing documentation - just straightforward requests that work.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link href="/docs" className="px-8 py-4 bg-gradient-to-r from-blue-500 to-cyan-500 text-white font-semibold rounded-lg hover:from-blue-600 hover:to-cyan-600 transition-all duration-300 transform">
                View Complete API Docs
              </Link>
            </div>
          </div>
        </div>
      </section>

      <footer className="relative z-10 px-6 py-20 border-t border-blue-800/30 bg-black">
        <div className="absolute inset-0 bg-gradient-to-br from-blue-900/10 to-cyan-900/5"></div>
        <div className="max-w-7xl mx-auto relative">
          <div className="grid md:grid-cols-4 gap-12 mb-16">
            <div className="md:col-span-2">
              <div className="flex items-center space-x-3 mb-6">

                <span className="text-xl font-bold font-[family-name:var(--font-unbounded)] bg-gradient-to-r from-blue-400 via-cyan-400 to-blue-400 bg-clip-text text-transparent">
                  Orbital AMM
                </span>
              </div>
              <p className="text-neutral-300 leading-relaxed mb-6 max-w-md font-[family-name:var(--font-spline-sans-mono)]">
                The future of AMM infrastructure is here. Unlock superior capital efficiency with spherical geometry and orbital mechanics for next-generation DeFi.
              </p>
            </div>

            <div>
              <h4 className="text-white font-semibold mb-6 font-[family-name:var(--font-unbounded)]">Product</h4>
              <div className="space-y-3">
                {[
                  { name: 'Swap', href: '/swap' },
                  { name: 'Liquidity', href: '/liquidity' },
                  { name: 'Documentation', href: '/docs' }
                ].map((item) => (
                  <Link key={item.name} href={item.href} className="block text-neutral-300 hover:text-blue-400 transition-colors duration-300 font-[family-name:var(--font-spline-sans-mono)]">
                    {item.name}
                  </Link>
                ))}
              </div>
            </div>

            <div>
              <h4 className="text-white font-semibold mb-6 font-[family-name:var(--font-unbounded)]">Resources</h4>
              <div className="space-y-3">
                {['Documentation', 'Tutorials', 'Blog', 'Support'].map((item) => (
                  <a key={item} href="#" className="block text-neutral-300 hover:text-blue-400 transition-colors duration-300 font-[family-name:var(--font-spline-sans-mono)]">
                    {item}
                  </a>
                ))}
              </div>
            </div>
          </div>

          <div className="border-t border-blue-800/30 pt-8 flex flex-col md:flex-row items-center justify-between">
            <p className="text-neutral-400 text-sm mb-4 md:mb-0 font-[family-name:var(--font-spline-sans-mono)]">
              © 2025 Orbital AMM. All rights reserved. Built for the future of finance.
            </p>
            <div className="flex space-x-6 text-sm">
              {['Privacy Policy', 'Terms of Service', 'Cookie Policy'].map((item) => (
                <a key={item} href="#" className="text-neutral-400 hover:text-blue-400 transition-colors duration-300 font-[family-name:var(--font-spline-sans-mono)]">
                  {item}
                </a>
              ))}
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Landing;