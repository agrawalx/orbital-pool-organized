'use client';

import React from 'react';
import { useAccount } from 'wagmi';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Clock } from 'lucide-react';

const OrderHistory: React.FC = () => {
  const { isConnected } = useAccount();

  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <Clock className="w-5 h-5 text-blue-400" />
          <span>Order History</span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {!isConnected ? (
          <p className="text-neutral-400 text-center py-8">
            Connect your wallet to view order history
          </p>
        ) : (
          <p className="text-neutral-400 text-center py-8">
            No orders found
          </p>
        )}
      </CardContent>
    </Card>
  );
};

export default OrderHistory;
