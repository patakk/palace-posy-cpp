#pragma once

#include "ofMain.h"
#include "ofxGui.h"

class ofApp : public ofBaseApp {
	public:
		void setup();
		void update();
		void draw();
		void reset();

		void keyPressed(int key);
		void keyReleased(int key);
		void mouseMoved(int x, int y);
		void mouseDragged(int x, int y, int button);
		void mousePressed(int x, int y, int button);
		void mouseReleased(int x, int y, int button);
		void windowResized(int w, int h);
		void dragEvent(ofDragInfo dragInfo);
		void gotMessage(ofMessage msg);

		ofFloatColor getRandomColor(float p);

		ofFbo layer;
		ofFbo baseLayer;
		ofFbo tempLayer;

		ofPixels pixs;

		ofShader blurShader;
		ofShader postShader;

		ofxPanel gui;
		ofxIntField gvariant;
		ofxIntField gseed;
		ofxIntField gscale;

		int variant;
		int seed;
		int sscale = 2;

		static const int DIMX = 1200;
		static const int DIMY = 1500;
		int DIMXS = DIMX * sscale;
		int DIMYS = DIMY * sscale;
		int flag = 0;

		int colori[3] = { 10, 13, 9 };
		int palettei = 0;

		float colors[31][3] = {
			{0.619, 0.522, 0.491},
			{0.939, 0.861, 0.348},
			{0.925, 0.601, 0.669},
			{0.830, 0.061, 0.071},
			{0.944, 0.621, 0.642},
			{0.941, 0.874, 0.323},
			{0.229, 0.447, 0.921},
			{0.248, 0.440, 0.955},
			{0.816, 0.032, 0.089},
			{0.630, 0.547, 0.485},

			{0.620, 1.000, 0.971},
			{0.951, 0.375, 0.230},
			{0.982, 0.776, 0.098},
			{0.258, 0.286, 0.435},
			{0.700, 0.768, 0.563},
			{0.524, 0.559, 0.518},
			{0.465, 0.756, 0.817},
			{0.661, 0.471, 0.465},
			{0.664, 0.942, 0.611},
			{0.222, 0.120, 0.288},
			{0.952, 0.348, 0.243},
			{0.688, 0.260, 0.300},

			{0.362, 0.514, 0.672},
			{0.719, 0.505, 0.511},
			{0.965, 0.972, 0.965},
			{0.925, 0.329, 0.336},
			{0.170, 0.276, 0.368},
			{0.000, 0.381, 0.523},
			{0.425, 0.462, 0.493},
			{0.945, 0.549, 0.517},
			{0.059, 0.072, 0.056},
		};
};