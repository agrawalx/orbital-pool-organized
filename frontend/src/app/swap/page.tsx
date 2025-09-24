'use client'
import React, { useState } from 'react';
import { useAccount } from 'wagmi';
import Navigation from '../../components/Navigation';
import { Card, CardContent, CardHeader, CardTitle } from '../../components/ui/Card';
import { Button } from '../../components/ui/Button';
import {
  ArrowUpDown,
} from 'lucide-react';

interface SwapState {
  fromTokenIndex: number;
  toTokenIndex: number;
  fromAmount: string;
  minAmountOut: string;
}

// const useSwap = () => {

// };


const OrbitalSwapPage = () => {
  const { /*address,*/isConnected } = useAccount();
  // const swapMutation = useSwap();

  const staticTokens = [
    { index: 0, symbol: 'USDC', address: '0x...' },
    { index: 1, symbol: 'USDT', address: '0x...' },
    { index: 2, symbol: 'DAI', address: '0x...' },
    { index: 3, symbol: 'FRAX', address: '0x...' },
    { index: 4, symbol: 'LUSD', address: '0x...' }
  ];

  const [swapState, setSwapState] = useState<SwapState>({
    fromTokenIndex: 0,
    toTokenIndex: 1,
    fromAmount: '',
    minAmountOut: '0'
  });

  const [isLoading, /*setIsLoading*/] = useState(false);

  const orbitalTokens = staticTokens;

  // const TokenIcon = ({ symbol, size = 24 }: { symbol: string; size?: number }) => {
  //   const getTokenColor = (symbol: string) => {
  //     switch (symbol) {
  //       case 'USDC': return 'from-blue-500 to-blue-600';
  //       case 'USDT': return 'from-green-500 to-green-600';
  //       case 'DAI': return 'from-yellow-500 to-yellow-600';
  //       case 'FRAX': return 'from-purple-500 to-purple-600';
  //       case 'LUSD': return 'from-red-500 to-red-600';
  //       default: return 'from-orange-500 to-red-500';
  //     }
  //   };

  //   return (
  //     <div className={`w-6 h-6 bg-gradient-to-br ${getTokenColor(symbol)} rounded-full flex items-center justify-center`}>
  //       <span className="text-white text-xs font-bold">{symbol.slice(0, 2)}</span>
  //     </div>
  //   );
  // };

  const handleSwap = () => {
    setSwapState(prev => ({
      ...prev,
      fromTokenIndex: prev.toTokenIndex,
      toTokenIndex: prev.fromTokenIndex,
      fromAmount: '',
      minAmountOut: '0'
    }));
  };

  const executeSwap = () => { }

  return (
    <div className="min-h-screen bg-black text-white">
      <Navigation />

      <div className="container mx-auto px-6 py-24">
        <div className="max-w-md mx-auto">
          <div className="text-center mb-8">
            <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent mb-2">
              ORBITAL SWAP
            </h1>
          </div>

          <Card className="bg-neutral-900/50 border-neutral-800 rounded-lg">
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span>Swap Tokens</span>
              </CardTitle>
            </CardHeader>

            <CardContent className="space-y-4">
              <div className="space-y-2">
                <label className="text-sm text-neutral-400">From</label>
                <div className="bg-neutral-800/50 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <select
                      value={swapState.fromTokenIndex}
                      onChange={(e) => setSwapState(prev => ({ ...prev, fromTokenIndex: parseInt(e.target.value) }))}
                      className="bg-transparent text-white font-semibold focus:outline-none"
                    >
                      {orbitalTokens.map((token) => (
                        <option key={token.index} value={token.index} className="bg-neutral-800">
                          {token.symbol}
                        </option>
                      ))}
                    </select>
                    <span className="text-neutral-400 text-sm">Balance: 0.00</span>
                  </div>
                  <input
                    type="number"
                    placeholder="0.0"
                    value={swapState.fromAmount}
                    onChange={(e) => setSwapState(prev => ({ ...prev, fromAmount: e.target.value }))}
                    className="w-full bg-transparent text-2xl font-semibold focus:outline-none"
                  />
                </div>
              </div>

              <div className="flex justify-center">
                <button
                  onClick={handleSwap}
                  className="p-2 bg-neutral-800 hover:bg-neutral-700 rounded-full transition-colors"
                >
                  <ArrowUpDown className="w-5 h-5" />
                </button>
              </div>

              <div className="space-y-2">
                <label className="text-sm text-neutral-400">To</label>
                <div className="bg-neutral-800/50 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <select
                      value={swapState.toTokenIndex}
                      onChange={(e) => setSwapState(prev => ({ ...prev, toTokenIndex: parseInt(e.target.value) }))}
                      className="bg-transparent text-white font-semibold focus:outline-none"
                    >
                      {orbitalTokens.map((token) => (
                        <option key={token.index} value={token.index} className="bg-neutral-800">
                          {token.symbol}
                        </option>
                      ))}
                    </select>
                    <span className="text-neutral-400 text-sm">Balance: 0.00</span>
                  </div>
                  <div className="text-2xl font-semibold text-neutral-300">
                    0.0
                  </div>
                </div>
              </div>

              <Button
                onClick={executeSwap}
                disabled={!isConnected || !swapState.fromAmount || isLoading || swapState.fromTokenIndex === swapState.toTokenIndex}
                className="w-full bg-gradient-to-r from-blue-500 rounded-md to-cyan-500 hover:from-blue-600 hover:to-cyan-600 disabled:opacity-50"
              >
                {!isConnected ? 'Connect Wallet' :
                  isLoading ? 'Swapping...' :
                    swapState.fromTokenIndex === swapState.toTokenIndex ? 'Select Different Tokens' :
                      'Swap Tokens'}
              </Button>
            </CardContent>
          </Card>

          <div className="mt-6 text-center text-sm text-neutral-500">
            <p>Pool Address: 0x83EC719A6F504583d0F88CEd111cB8e8c0956431</p>
            <p>Network: Arbitrum Sepolia</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default OrbitalSwapPage;
