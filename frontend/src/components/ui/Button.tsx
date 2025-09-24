import React from 'react';
import { cn } from "../../lib/utils"

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'default' | 'destructive' | 'outline' | 'secondary' | 'ghost' | 'link' | 'gradient';
  size?: 'default' | 'sm' | 'lg' | 'icon';
  children: React.ReactNode;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = 'default', size = 'default', children, ...props }, ref) => {
    return (
      <button
        className={cn(
          'inline-flex items-center justify-center rounded-2xl text-sm font-medium transition-all duration-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 active:scale-[0.98] font-[family-name:var(--font-unbounded)]',
          {
            'bg-neutral-900/60 text-neutral-50 hover:bg-neutral-800/80 shadow-lg border border-neutral-700/50 backdrop-blur-sm': variant === 'default',
            'bg-red-600 text-neutral-50 hover:bg-red-700 shadow-lg': variant === 'destructive',
            'border border-neutral-600/50 bg-transparent hover:bg-neutral-800/30 hover:text-neutral-50 backdrop-blur-sm': variant === 'outline',
            'bg-neutral-800/50 text-neutral-50 hover:bg-neutral-700/60 backdrop-blur-sm': variant === 'secondary',
            'hover:bg-neutral-800/30 hover:text-neutral-50': variant === 'ghost',
            'text-blue-400 underline-offset-4 hover:underline hover:text-cyan-300': variant === 'link',
            'bg-gradient-to-r from-blue-500 to-cyan-500 text-white hover:from-blue-600 hover:to-cyan-600 shadow-lg shadow-blue-500/25 hover:scale-105': variant === 'gradient',
          },
          {
            'h-10 px-4 py-2': size === 'default',
            'h-9 rounded-lg px-3 text-xs': size === 'sm',
            'h-12 rounded-2xl px-8 text-base': size === 'lg',
            'h-10 w-10': size === 'icon',
          },
          className
        )}
        ref={ref}
        {...props}
      >
        {children}
      </button>
    );
  }
);

Button.displayName = 'Button';

// Export buttonVariants for other components that need it
export const buttonVariants = {
  default: 'bg-neutral-900/60 text-neutral-50 hover:bg-neutral-800/80 shadow-lg border border-neutral-700/50 backdrop-blur-sm',
  destructive: 'bg-red-600 text-neutral-50 hover:bg-red-700 shadow-lg',
  outline: 'border border-neutral-600/50 bg-transparent hover:bg-neutral-800/30 hover:text-neutral-50 backdrop-blur-sm',
  secondary: 'bg-neutral-800/50 text-neutral-50 hover:bg-neutral-700/60 backdrop-blur-sm',
  ghost: 'hover:bg-neutral-800/30 hover:text-neutral-50',
  link: 'text-blue-400 underline-offset-4 hover:underline hover:text-cyan-300',
  gradient: 'bg-gradient-to-r from-blue-500 to-cyan-500 text-white hover:from-blue-600 hover:to-cyan-600 shadow-lg shadow-blue-500/25 hover:scale-105'
};

export { Button }; 