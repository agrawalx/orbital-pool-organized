// SPDX-License-Identifier: MIT
pragma solidity ^0.8.30;

import {Script, console} from "forge-std/Script.sol";
import {OrbitalPool} from "../src/Orbital.sol";
import {OrbitalFactory} from "../src/OrbitalFactory.sol";
import {MockUSDC} from "../src/MockUSDC.sol";
import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";

/**
 * @title DeployAndTest
 * @notice Complete deployment and testing script for OrbitalPool via OrbitalFactory
 * @dev Deploys tokens, factory, pool via factory, approves, adds liquidity, and executes swap
 */
contract DeployAndTest is Script {
        // Constants
    address constant MATH_HELPER = 0xe7Ec8b37FC4EDc703003561d67241583C507BD9b;
    uint256 constant TOKENS_COUNT = 5;
    uint256 constant PRECISION = 1e15; // Use 1e15 precision as requested
    uint256 constant SQRT5_SCALED = 2236067977499790; // sqrt(5) * 1e15
    uint256 constant Q96X48_SCALE = 2**48; // Scaling factor for Q96.48 format
    uint256 constant MILLION_TOKENS = 1_000_000 * 1e18; // 1 million tokens with 18 decimals
    
    // Deployed contracts
    OrbitalFactory public orbitalFactory;
    OrbitalPool public orbitalPool;
    MockUSDC[TOKENS_COUNT] public mockTokens;
    
    function run() external {
        vm.startBroadcast();
        
        console.log("=== ORBITAL POOL DEPLOYMENT AND TESTING ===");
        console.log("Math Helper Address:", MATH_HELPER);
        console.log("Deployer:", msg.sender);
        
        // Step 1: Deploy MockUSDC tokens
        _deployTokens();
        
        // Step 2: Deploy OrbitalFactory
        _deployFactory();
        
        // Step 3: Deploy OrbitalPool using the factory
        _deployPool();
        console.log("pool deployed"); 
        // Step 4: Approve tokens for the pool
        _approveTokens();
        
        // Step 5: Add liquidity multiple times with different p values and reserves
        // _addLiquidityMultipleTimes();
        
        console.log("=== DEPLOYMENT AND TESTING COMPLETED ===");
        
        vm.stopBroadcast();
    }
    
    function _deployTokens() internal {
        console.log("\n--- DEPLOYING MOCK USDC TOKENS ---");
        
        for (uint256 i = 0; i < TOKENS_COUNT; i++) {
            string memory name = string(abi.encodePacked("Mock USDC ", _getTokenLetter(i)));
            string memory symbol = string(abi.encodePacked("MUSDC-", _getTokenLetter(i)));
            
            mockTokens[i] = new MockUSDC(name, symbol, MILLION_TOKENS / 1e18);
            
            console.log("Token", i, "deployed:", address(mockTokens[i]));
            console.log("  Name:", name);
            console.log("  Symbol:", symbol);
            console.log("  Initial supply: 1,000,000 tokens");
        }
    }
    
    function _deployFactory() internal {
        console.log("\n--- DEPLOYING ORBITAL FACTORY ---");
        
        orbitalFactory = new OrbitalFactory(MATH_HELPER);
        
        console.log("OrbitalFactory deployed:", address(orbitalFactory));
        console.log("Math Helper Address:", orbitalFactory.MATH_HELPER_ADDRESS());
    }
    
    function _deployPool() internal {
        console.log("\n--- DEPLOYING ORBITAL POOL VIA FACTORY ---");
        
        // Create IERC20 dynamic array from deployed mock tokens
        IERC20[] memory tokens = new IERC20[](TOKENS_COUNT);
        for (uint256 i = 0; i < TOKENS_COUNT; i++) {
            tokens[i] = IERC20(address(mockTokens[i]));
        }
        
        // Deploy pool using the factory
        address poolAddress = orbitalFactory.createPool(tokens);
        orbitalPool = OrbitalPool(poolAddress);
        
        console.log("OrbitalPool deployed:", address(orbitalPool));
        console.log("Math Helper:", address(orbitalPool.MATH_HELPER()));
        console.log("Tokens Count:", orbitalPool.TOKENS_COUNT());
        
        // Verify factory records
        console.log("Factory pool count:", orbitalFactory.getPoolCount());
    }
    
    function _approveTokens() internal {
        console.log("\n--- APPROVING TOKENS ---");
        
        uint256 approveAmount = MILLION_TOKENS; // Approve 1 million tokens each
        
        for (uint256 i = 0; i < TOKENS_COUNT; i++) {
            mockTokens[i].approve(address(orbitalPool), approveAmount);
            console.log("Approved token", i, "amount:", approveAmount / 1e18);
            
            // Verify approval
            uint256 allowance = mockTokens[i].allowance(msg.sender, address(orbitalPool));
            console.log("  Verified allowance:", allowance / 1e18);
        }
    }
    
    function _addLiquidityMultipleTimes() internal {
        console.log("\n--- ADDING LIQUIDITY MULTIPLE TIMES ---");
        
        // p values between 0.80 and 0.99 (scaled by 2**48)
        uint144[10] memory pValues = [
            uint144((80 * Q96X48_SCALE) / 100),   // 0.80
            uint144((82 * Q96X48_SCALE) / 100),   // 0.82
            uint144((84 * Q96X48_SCALE) / 100),   // 0.84
            uint144((86 * Q96X48_SCALE) / 100),   // 0.86
            uint144((88 * Q96X48_SCALE) / 100),   // 0.88
            uint144((90 * Q96X48_SCALE) / 100),   // 0.90
            uint144((92 * Q96X48_SCALE) / 100),   // 0.92
            uint144((94 * Q96X48_SCALE) / 100),   // 0.94
            uint144((96 * Q96X48_SCALE) / 100),   // 0.96
            uint144((99 * Q96X48_SCALE) / 100)    // 0.99
        ];
        
        // Reserve amounts from 10,000 to 100,000 (scaled by 2**48)
        for (uint256 i = 0; i < 10; i++) {
            uint256 baseAmount = 10000 + (i * 10000); // 10k, 20k, 30k, ..., 100k
            uint144 scaledAmount = uint144((baseAmount * 1e18 * Q96X48_SCALE) / 1e18); // Convert to Q96.48
            
            // Create amounts array with same amount for each token
            uint144[] memory amounts = new uint144[](TOKENS_COUNT);
            for (uint256 j = 0; j < TOKENS_COUNT; j++) {
                amounts[j] = scaledAmount;
            }
            
            console.log("\n--- LIQUIDITY ADDITION", i + 1, "---");
            console.log("p value (scaled):", uint256(pValues[i]));
            console.log("p value (decimal):", (uint256(pValues[i]) * 100) / Q96X48_SCALE);
            console.log("Base amount per token:", baseAmount);
            console.log("Scaled amount per token:", uint256(scaledAmount));
            
            try orbitalPool.addLiquidity(pValues[i], amounts) {
                console.log("SUCCESS: Liquidity added successfully");
                
                // Log some pool state
                uint144 tickRadius = orbitalPool.getTickRadius(pValues[i]);
                uint144 tickLiquidity = orbitalPool.getTickLiquidity(pValues[i]);
                console.log("  Tick radius:", uint256(tickRadius));
                console.log("  Tick liquidity:", uint256(tickLiquidity));
                
            } catch Error(string memory reason) {
                console.log("FAILED: Failed to add liquidity:", reason);
            } catch {
                console.log("FAILED: Failed to add liquidity: Unknown error");
            }
        }
    }
    

    function _getTokenLetter(uint256 index) internal pure returns (string memory) {
        if (index == 0) return "A";
        if (index == 1) return "B";
        if (index == 2) return "C";
        if (index == 3) return "D";
        if (index == 4) return "E";
        return "X";
    }
    
    // Utility function to get deployed addresses
    function getDeployedAddresses() external view returns (
        address factoryAddress,
        address poolAddress,
        address[] memory tokenAddresses
    ) {
        factoryAddress = address(orbitalFactory);
        poolAddress = address(orbitalPool);
        tokenAddresses = new address[](TOKENS_COUNT);
        for (uint256 i = 0; i < TOKENS_COUNT; i++) {
            tokenAddresses[i] = address(mockTokens[i]);
        }
    }
}
