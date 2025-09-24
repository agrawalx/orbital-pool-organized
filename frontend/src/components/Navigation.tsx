'use client'
import React, { useState } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { usePathname } from 'next/navigation';
import { 
  Menu, 
  X, 
} from 'lucide-react';
import { CustomConnectButton, MobileCustomConnectButton } from './CustomConnectButton';

const Navigation: React.FC = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const pathname = usePathname();

  const navItems = [
    { href: '/', label: 'Home' },
    { href: '/swap', label: 'Swap' },
    { href: '/liquidity', label: 'Liquidity' },
    { href: '/docs', label: 'Docs' },
  ];

  const isActive = (href: string) => pathname === href;

  return (
    <nav className="px-4 py-2 bg-transparent sticky top-0 z-50">
      <div className="max-w-8xl mx-auto flex items-center justify-between">
        <Link href="/" className="group cursor-pointer">
          <Image 
            src="/logo.png" 
            alt="Orbital AMM Logo" 
            width={2500}
            height={2500}
            className="h-20 w-auto object-contain scale-300 ml-12 transition-all duration-300 py-4"
            priority
          />
        </Link>

        <div className="hidden md:flex items-center space-x-8">
          {navItems.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className={`relative px-6 py-3 rounded-xl text-sm font-medium transition-all duration-300 flex items-center space-x-2 ${
                isActive(item.href)
                  ? 'text-white bg-neutral-800/50 border border-neutral-700/50'
                  : 'text-neutral-300 hover:text-white hover:bg-neutral-800/30'
              }`}
            >
              <span>{item.label}</span>
              {isActive(item.href) && (
                <div className="absolute bottom-0 left-0 w-full h-0.5 bg-gradient-to-r from-blue-400 to-cyan-400 rounded-full"></div>
              )}
            </Link>
          ))}
        </div>

        <div className="hidden md:flex items-center space-x-4">
          <CustomConnectButton />
        </div>

        <button
          className="md:hidden p-2 rounded-sm bg-neutral-800/50 hover:bg-neutral-800/70 rounded-xl transition-all duration-300 backdrop-blur-sm border border-neutral-700/50"
          onClick={() => setIsMenuOpen(!isMenuOpen)}
        >
          {isMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
        </button>
      </div>

      {isMenuOpen && (
        <div className="md:hidden absolute top-full left-0 right-0 bg-neutral-900/95 backdrop-blur-xl border-b border-neutral-800/50 animate-slideDown">
          <div className="px-6 py-6 space-y-4">
            {navItems.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className={`flex items-center space-x-3 px-4 py-3 rounded-xl transition-all duration-300 font-[family-name:var(--font-unbounded)] ${
                  isActive(item.href)
                    ? 'text-white bg-neutral-800/50 border border-neutral-700/50'
                    : 'text-neutral-300 hover:text-white hover:bg-neutral-800/30'
                }`}
                onClick={() => setIsMenuOpen(false)}
              >
                <span>{item.label}</span>
              </Link>
            ))}
            
            <div className="pt-4 border-t border-neutral-800/50 space-y-3">
              <MobileCustomConnectButton />
            </div>
          </div>
        </div>
      )}
    </nav>
  );
};

export default Navigation; 