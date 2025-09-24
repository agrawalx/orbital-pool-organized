'use client'
import React, { useState, useEffect } from 'react';
import { useAccount, useWriteContract, useWaitForTransactionReceipt } from 'wagmi';
import Navigation from '../../components/Navigation';
import { Card, CardContent, CardHeader, CardTitle } from '../../components/ui/Card';
import { Button } from '../../components/ui/Button';
import { Info, Layers } from 'lucide-react';

interface LiquidityState {
  amounts: string[];
  tolerance: string;
  mode: 'add' | 'remove';
  kValue: string;
}

const OrbitalLiquidityPage = () => {
  const { /*address,*/ isConnected } = useAccount();
  const { /*writeContract,*/ data: hash, isPending } = useWriteContract();
  const { isLoading: isConfirming, isSuccess } = useWaitForTransactionReceipt({
    hash,
  });

  const [liquidityState, setLiquidityState] = useState<LiquidityState>({
    amounts: ['', '', '', '', ''],
    tolerance: '',
    mode: 'add',
    kValue: '0' 
  });

  const orbitalTokens =[
    { index: 0, symbol: 'USDT', address: '0x...' },
    { index: 1, symbol: 'USDC', address: '0x...' },
    { index: 2, symbol: 'DAI', address: '0x...' },
    { index: 3, symbol: 'FRAX', address: '0x...' },
    { index: 4, symbol: 'LUSD', address: '0x...' }
  ]

  const TokenIcon = ({ symbol }: { symbol: string }) => {
    const getTokenColor = (symbol: string) => {
      switch (symbol) {
        case 'USDC': return 'from-blue-500 to-blue-600';
        case 'USDT': return 'from-green-500 to-green-600';
        case 'DAI': return 'from-yellow-500 to-yellow-600';
        case 'FRAX': return 'from-purple-500 to-purple-600';
        case 'LUSD': return 'from-red-500 to-red-600';
        default: return 'from-orange-500 to-red-500';
      }
    };

    return (
      <div className={`w-6 h-6 bg-gradient-to-br ${getTokenColor(symbol)} rounded-full flex items-center justify-center`}>
        <span className="text-white text-xs font-bold">{symbol.slice(0, 2)}</span>
      </div>
    );
  };

  const executeAddLiquidity = () => {}

  const handleAmountChange = (value: string) => {
    let sanitized = value.replace(/[^0-9]/g, '');
    sanitized = sanitized.replace(/^0+(\d)/, '$1');
    const newAmounts = Array(5).fill(sanitized);
    setLiquidityState(prev => ({ ...prev, amounts: newAmounts }));
  };

  const handleToleranceChange = (value: string) => {
    let sanitized = value.replace(/[^0-9.]/g, '');
    const parts = sanitized.split('.');
    if (parts.length > 2) {
      sanitized = parts[0] + '.' + parts.slice(1).join('');
    }
    setLiquidityState(prev => ({ ...prev, tolerance: sanitized }));
  };

  
  useEffect(() => {
    if (isSuccess) {
      setLiquidityState((prev: LiquidityState) => ({
        ...prev,
        amounts: ['', '', '', '', ''],
        tolerance: ''
      }));
      alert(`Liquidity added successfully! Transaction: ${hash}`);
    }
  }, [isSuccess, hash]);

  const hasAmounts = liquidityState.amounts.some((amt: string) => parseInt(amt || '0') > 0);
  const hasTolerance = liquidityState.tolerance && !isNaN(Number(liquidityState.tolerance)) && Number(liquidityState.tolerance) > 0;
  const isProcessing = isPending || isConfirming;

  return (
    <div className="min-h-screen bg-black text-white">
      <Navigation />

      <div className="container mx-auto px-6 py-24">
        <div className="max-w-2xl mx-auto">
          <div className="text-center mb-8">
            <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent mb-2">
              Liquidity
            </h1>
          </div>

          <div className="flex bg-neutral-900/50 rounded-lg p-1 mb-6">
            <button
              onClick={() => setLiquidityState(prev => ({ ...prev, mode: 'add' }))}
              className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors ${liquidityState.mode === 'add'
                ? 'bg-blue-500 text-white'
                : 'text-neutral-400 hover:text-white'
                }`}
            >
              Add Liquidity
            </button>
            <button
              onClick={() => setLiquidityState(prev => ({ ...prev, mode: 'remove' }))}
              className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors ${liquidityState.mode === 'remove'
                ? 'bg-blue-500 text-white'
                : 'text-neutral-400 hover:text-white'
                }`}
            >
              Remove Liquidity
            </button>
          </div>

          <Card className="bg-neutral-900/50 border-neutral-800">
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Layers className="w-5 h-5 text-blue-400" />
                <span>{liquidityState.mode === 'add' ? 'Add' : 'Remove'} Liquidity</span>
              </CardTitle>
            </CardHeader>

            <CardContent className="space-y-6">
              <div className="space-y-4">
                <label className="text-sm text-neutral-400">
                  Amount (same for all 5 tokens)
                </label>
                <div className="bg-neutral-800/50 rounded-lg p-4">
                  <input
                    type="text"
                    inputMode="numeric"
                    pattern="[0-9]*"
                    placeholder="0"
                    value={liquidityState.amounts[0] || ''}
                    onChange={(e) => handleAmountChange(e.target.value)}
                    className="w-full bg-transparent text-2xl font-semibold focus:outline-none text-center"
                  />
                  <p className="text-center text-sm text-neutral-500 mt-2">
                    This amount will be applied to all 5 tokens
                  </p>
                </div>
              </div>

              <div className="space-y-3">
                <label className="text-sm text-neutral-400">Tokens</label>
                {orbitalTokens.map((token, index) => (
                  <div key={token.index} className="flex items-center justify-between bg-neutral-800/30 rounded-lg p-3">
                    <div className="flex items-center space-x-3">
                      <TokenIcon symbol={token.symbol} />
                      <span className="font-semibold">{token.symbol}</span>
                    </div>
                    <div className="text-right">
                      <div className="font-semibold">{liquidityState.amounts[index] || '0'}</div>
                      <div className="text-xs text-neutral-500">Balance: 0.00</div>
                    </div>
                  </div>
                ))}
              </div>

              <div className="space-y-2">
                <label className="text-sm text-neutral-400">Tolerance (%)</label>
                <div className="flex space-x-2">
                  <input
                    type="text"
                    inputMode="decimal"
                    pattern="^\d*\.?\d*$"
                    value={liquidityState.tolerance}
                    onChange={(e) => handleToleranceChange(e.target.value)}
                    className="flex-1 px-3 py-1 bg-neutral-800 rounded text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
                    placeholder="Enter tolerance (e.g. 0.5)"
                  />
                </div>
                <p className="text-xs text-neutral-500">
                  Enter your desired slippage tolerance as a percentage (e.g. 0.5)
                </p>
              </div>

              <Button
                onClick={executeAddLiquidity}
                disabled={!isConnected || !hasAmounts || !hasTolerance || isProcessing}
                className="w-full bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 disabled:opacity-50"
              >
                {!isConnected ? 'Connect Wallet' :
                  isProcessing ?
                    (isPending ? 'Confirming...' : 'Processing Transaction...') :
                    !hasAmounts ? 'Enter Amount' :
                      !hasTolerance ? 'Enter Tolerance' :
                        `${liquidityState.mode === 'add' ? 'Add' : 'Remove'} Liquidity`}
              </Button>

              <div className="bg-neutral-800/30 rounded-lg p-4">
                <div className="flex items-start space-x-2">
                  <Info className="w-4 h-4 text-blue-400 mt-0.5" />
                  <div className="text-sm text-neutral-300">
                    <p className="font-semibold mb-1">Important Notes:</p>
                    <ul className="text-xs text-neutral-400 space-y-1">
                      <li>• All 5 tokens must have the same integer amount</li>
                      <li>• You&apos;ll receive LP tokens representing your pool share</li>
                      <li>• Tolerance helps protect against price changes</li>
                      <li>• You earn fees proportional to your liquidity share</li>
                    </ul>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <div className="mt-6 text-center text-sm text-neutral-500">
            <p>Pool Address: 0x...</p>
            <p>Network: Arbitrum Sepolia</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default OrbitalLiquidityPage;
