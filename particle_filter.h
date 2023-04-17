#ifndef __PATICLE_FILTER_H
#define __PARICLE_FILTER_H

#include <iostream>
#include <Eigen/Dense>

#include <tuple>

#include <vector>

#include <random>


namespace pf
{


template<typename T, int StateDimension>
using StateType = typename Eigen::Matrix<T, StateDimension, 1>;

template<typename T, int MeasuremantDimension>
using MeasurementType = typename Eigen::Matrix<T, MeasuremantDimension, 1>;

template<typename T, int StateDimension>
class Particle
{
public:
	using DataType = T;
	using StateVector = StateType<T, StateDimension>;

	Particle()
	{

	}

	Particle( const StateVector& state ) : state_( state )
	{

	}

	~Particle()
	{

	}	

	const StateVector& getState() const
	{
		return state_;
	}

	const DataType& getWeight() const
	{
		return weight_;
	}

	StateVector state_ = StateVector::Zero();
	DataType weight_ = 0;

};

// CRTP Base Class
//template<typename DerivedType, typename T, int StateDimension>

template<template<typename U, int Dimension> class Derived, typename T, int StateDimension>
class ParticleFilterBase
{
public:
	using DerivedType = Derived<T, StateDimension>;

	ParticleFilterBase()
	{

	}

	virtual ~ParticleFilterBase()
	{

	}

	// 1. interface 1 : particles initialization
	void initializeParticles( const StateType<T, StateDimension>& initial_state ) 
	{
		if ( auto ptr = static_cast<DerivedType*>( this ) ) {
			ptr->initializeParticles();
		}
		
		is_initialized_ = true;
	}

	// 2. interface 2 : sample from state transition function
	template<typename ...ParasType>
	void predict( ParasType&&... paras )
	{
		if ( auto ptr = static_cast<DerivedType*>( this ) ) {
			ptr->predict( paras... );
		}	
	}

	// 3. interface 3 : update weight of every particle  from measurement function 
	template<typename ...ParasType>
	void update( ParasType&&... paras ) 
	{
		if ( auto ptr = static_cast<DerivedType*>( this ) ) {
			ptr->update( paras... );
		}
	}

	// 3.1 nomalize the weights	

	// 4. interface 4 : resample 
		

	// 5. 

protected:
	StateType<T, StateDimension> generateNormalDistribution( StateType<T, StateDimension>&& initial_state, 
					 T sigma_ )
	{
		std::default_random_engine gen;
		StateType<T, StateDimension> ret_state = StateType<T, StateDimension>::Zero();

		for ( size_t i = 0; i < StateDimension; i ++ ) {
			std::normal_distribution<T> normal_dist( initial_state[i], sigma_ );
			ret_state[i] = normal_dist( gen );
		
		}
	}	

	void setParticlesNumber( const int num )
	{
		particles_.resize( num );
	}

	const int getParticlesNumber() const
	{
		return particles_.size();
	}

protected:
	bool is_initialized_ = false;

	std::vector<Particle<T, StateDimension>> particles_;
};






}

#endif
