#include "ofMain.h"
#include "ofApp.h"

//========================================================================
int main() {

	ofGLWindowSettings settings;
	settings.setGLVersion(3, 2);
	settings.setSize(ofApp::DIMX, ofApp::DIMY);

	auto window = ofCreateWindow(settings);

	ofRunApp(window, make_shared<ofApp>());
	ofRunMainLoop();

}