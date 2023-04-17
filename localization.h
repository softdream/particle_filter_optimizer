#ifndef __LOCALIZATION_H
#define __LOCALIZATION_H

#include "particle_filter.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace localization
{

using namespace pf;

template<typename T, int Dimension = 3>
class Localization : public ParticleFilterBase<Localization, T, Dimension>
{
public:
	using StateVector = StateType<T, Dimension>;


	// 1. interface 1 : particles initialization
	void initializeParticles( const StateVector& initial_state )
	{
		this->setParticlesNumber( 100 );
		
		for ( auto& particle : this->particles_ ) {
			particle.state_ = generateNormalDistribution( initial_state, sigma_ );

			particle.weight_ = 1;
		}
	}
	
	// 2. interface 2 : sample from state transition function
	template<typename ...ParasType>
        void predict( ParasType&&... paras )
        {
		static_assert( !( sizeof...( paras ) != 2 ), "The number of Parameters is too few or too much !" );

		std::tuple<ParasType ...> tuple( paras... );
		
		// for pose of each particle, sample from the motion model 
		for ( auto& particle : this->particles_ ) {
			sampleOdometryMotionModel( particle.state_, std::get<0>( tuple ), std::get<1>( tuple ) );
		}
		
	}

	// 3. interface 3 : update weight of every particle  from measurement function 
        template<typename ...ParasType>
        void update( ParasType&&... paras )
        {
	
	}

private:
	void sampleOdometryMotionModel( StateVector& p, const StateVector& p_new, const StateVector& p_old )
	{
		auto delta_rot1 = ::atan2( p_new(1) - p_old(1), p_new(0) - p_old(0) ) - p_old(2);
		auto delta_trans = ::sqrt( ( p_old(0) - p_new(0) ) * ( p_old(0) - p_new(0) ) + ( p_old(1) - p_new(1) ) * ( p_old(1) - p_new(1) ) );
		auto delta_rot2 = p_new(2) - p_old(2) - delta_rot1;
		
		auto delta_rot1_hat = delta_rot1 - sampleGussian( alpha1_ * delta_rot1 + alpha2_ * delta_rot2 );
		auto delta_trans_hat = delta_trans - sampleGussian( alpha3_ * delta_trans + alpha4_ * (delta_rot1 + delta_rot2) );
		auto delta_rot2_hat = delta_rot2 - sampleGussian( alpha1_ * delta_rot2 + alpha2_ * delta_trans );

		p(0) += delta_trans_hat * ::cos( p(2) + delta_rot1_hat );
		p(1) += delta_trans_hat * ::sin( p(2) + delta_rot1_hat );
		p(2) += delta_rot1_hat + delta_rot2_hat;

		angleNormalize( p(2) );
	}

	const T sampleGussian( const T sigma )
	{
		std::default_random_engine gen;
		std::normal_distribution<T> normal_dist( 0, sigma );
		
		return normal_dist( gen );	
	}

	void angleNormalize( T&& angle )
	{
		if ( angle >= M_PI ) angle -= 2 * M_PI;
		
		if ( angle <= -M_PI ) angle += 2 * M_PI;
	}

private:
 	const T sigma_ = 1.0;

	// parameters for motion model
        const T alpha1_ = 0.025f, alpha2_ = 0.025f, alpha3_ = 0.4f, alpha4_ = 0.4f;

};

}

#endif
