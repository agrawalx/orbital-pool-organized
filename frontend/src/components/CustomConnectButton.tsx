'use client';

import React from 'react';
import { Wallet, ChevronDown } from 'lucide-react';
import { Button } from './ui/Button';
import { useAccount, useConnect, useDisconnect, useSwitchChain } from 'wagmi';
import { arbitrumSepolia } from 'wagmi/chains';

export const CustomConnectButton = () => {
  const { address, chainId, isConnected } = useAccount();
  const { connect, connectors, isPending: isConnecting } = useConnect();
  const { disconnect } = useDisconnect();
  const { switchChain, isPending: isSwitching } = useSwitchChain();

  const injectedConnector = connectors.find((c) => c.id === 'injected') || connectors[0];
  const shortAddr = address ? `${address.slice(0, 6)}...${address.slice(-4)}` : '';
  const wrongNetwork = chainId !== arbitrumSepolia.id;

  if (!isConnected) {
    return (
      <Button
        onClick={() => connect({ connector: injectedConnector })}
        disabled={isConnecting}
        variant="gradient"
        className="flex items-center space-x-2 font-[family-name:var(--font-unbounded)]"
      >
        <Wallet className="w-4 h-4" />
        <span>{isConnecting ? 'Connecting...' : 'Connect Wallet'}</span>
      </Button>
    );
  }

  return (
    <div className="flex items-center space-x-2">
      <Button
        onClick={() => wrongNetwork && switchChain({ chainId: arbitrumSepolia.id })}
        disabled={isSwitching}
        variant={wrongNetwork ? 'destructive' : 'secondary'}
        className="flex items-center rounded-md space-x-2 px-3 py-2 bg-neutral-800/50 hover:bg-neutral-700/50 border border-neutral-700/50 font-[family-name:var(--font-unbounded)]"
      >
        <span className="text-xs">{wrongNetwork ? 'Wrong network' : 'Arbitrum Sepolia'}</span>
        {wrongNetwork && <ChevronDown className="w-3 h-3" />}
      </Button>

      <Button
        onClick={() => disconnect()}
        variant="secondary"
        className="flex items-center space-x-2 px-3 rounded-md py-2 bg-neutral-800/50 hover:bg-neutral-700/50 border border-neutral-700/50 font-[family-name:var(--font-unbounded)]"
      >
        <Wallet className="w-4 h-4" />
        <span className="text-sm">{shortAddr}</span>
      </Button>
    </div>
  );
};

export const MobileCustomConnectButton = () => {
  const { address, chainId, isConnected } = useAccount();
  const { connect, connectors, isPending: isConnecting } = useConnect();
  const { disconnect } = useDisconnect();
  const { switchChain, isPending: isSwitching } = useSwitchChain();

  const injectedConnector = connectors.find((c) => c.id === 'injected') || connectors[0];
  const shortAddr = address ? `${address.slice(0, 6)}...${address.slice(-4)}` : '';
  const wrongNetwork = chainId !== arbitrumSepolia.id;

  if (!isConnected) {
    return (
      <Button
        onClick={() => connect({ connector: injectedConnector })}
        disabled={isConnecting}
        variant="gradient"
        className="w-full flex items-center justify-center rounded-md space-x-2 font-[family-name:var(--font-unbounded)]"
      >
        <Wallet className="w-4 h-4" />
        <span>{isConnecting ? 'Connecting...' : 'Connect Wallet'}</span>
      </Button>
    );
  }

  return (
    <div className="w-full space-y-2">
      <Button
        onClick={() => wrongNetwork && switchChain({ chainId: arbitrumSepolia.id })}
        disabled={isSwitching}
        variant={wrongNetwork ? 'destructive' : 'secondary'}
        className="w-full flex items-center rounded-md justify-center space-x-2 px-3 py-2 bg-neutral-800/50 hover:bg-neutral-700/50 border border-neutral-700/50 font-[family-name:var(--font-unbounded)]"
      >
        <span>{wrongNetwork ? 'Wrong network' : 'Arbitrum Sepolia'}</span>
      </Button>
      <Button
        onClick={() => disconnect()}
        variant="secondary"
        className="w-full flex items-center rounded-md justify-center space-x-2 px-3 py-2 bg-neutral-800/50 hover:bg-neutral-700/50 border border-neutral-700/50 font-[family-name:var(--font-unbounded)]"
      >
        <Wallet className="w-4 h-4" />
        <span className="text-sm">{shortAddr}</span>
      </Button>
    </div>
  );
}; 