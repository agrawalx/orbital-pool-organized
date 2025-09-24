import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';
import { http, createConfig } from 'wagmi';
import { arbitrumSepolia } from 'wagmi/chains';
import { injected } from 'wagmi/connectors';

export const config = createConfig({
  chains: [arbitrumSepolia],
  connectors: [injected()],
  transports: {
    [arbitrumSepolia.id]: http(),
  },
  ssr: true,
});

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}