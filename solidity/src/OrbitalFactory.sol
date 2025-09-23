// SPDX-License-Identifier: MIT
pragma solidity 0.8.30;

import {OrbitalPool} from "./Orbital.sol";
import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";

/**
 * @title OrbitalFactory
 * @notice Factory contract for deploying Orbital AMM pools
 * @dev Creates new OrbitalPool instances with configurable token counts
 * @author Orbital Protocol
 */
contract OrbitalFactory {
    address public immutable MATH_HELPER_ADDRESS;
    address[] public pools;
    mapping(address => PoolInfo) public poolInfo;

    event PoolCreated(
        address indexed pool,
        address[] tokens,
        uint256 tokenCount,
        address creator
    );

    /**
     * @notice Information about a deployed pool
     */
    struct PoolInfo {
        address[] tokens;
        uint256 tokenCount;
    }

    /**
     * @notice Initializes the factory
     */
    constructor(address mathHelperAddress) {
        require(
            mathHelperAddress != address(0),
            "Invalid math helper address"
        );
        MATH_HELPER_ADDRESS = mathHelperAddress;
    }

    /**
     * @notice Creates a new Orbital pool
     * @param tokens Array of ERC20 token addresses for the pool
     * @return pool Address of the newly created pool
     */
    function createPool(
        IERC20[] memory tokens
    ) external returns (address pool) {
        require(tokens.length > 0, "At least one token required");

        for (uint256 i = 0; i < tokens.length; i++) {
            require(address(tokens[i]) != address(0), "Invalid token address");
        }

        pool = address(new OrbitalPool(tokens, MATH_HELPER_ADDRESS));

        require(pool != address(0), "Pool creation failed");

        // Store pool info
        address[] memory tokenAddresses = new address[](tokens.length);
        for (uint256 i = 0; i < tokens.length; i++) {
            tokenAddresses[i] = address(tokens[i]);
        }

        poolInfo[pool] = PoolInfo({
            tokens: tokenAddresses,
            tokenCount: tokens.length
        });

        pools.push(pool);

        emit PoolCreated(pool, tokenAddresses, tokens.length, msg.sender);

        return pool;
    }

    /**
     * @notice Gets the number of deployed pools
     * @return Number of pools created by this factory
     */
    function getPoolCount() external view returns (uint256) {
        return pools.length;
    }

    /**
     * @notice Gets all deployed pool addresses
     * @return Array of pool addresses
     */
    function getAllPools() external view returns (address[] memory) {
        return pools;
    }
}